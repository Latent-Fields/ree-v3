#!/opt/local/bin/python3
"""
V3-EXQ-894a -- MECH-074d BLA PE-remap attribution selectivity, RECALIBRATED-THRESHOLD
               RETEST (PE-sigma SWEEP; WALL-INDEPENDENT, representation-level DV).

SUPERSEDES V3-EXQ-894 (run_id v3_exq_894_mech074d_bla_remap_attribution_selectivity_
20260808T005219Z_v3, FAIL/weakens). Routed from the confirmed autopsy
``failure_autopsy_V3-EXQ-894_2026-08-08`` (/failure-autopsy -> /queue-experiment,
same-claim retest). This driver is 894 with ONE design change: the single ON arm at the
SD-035 default sigma=1.0 becomes a SWEEP of PE-sigma thresholds. Everything else -- the
DV, the matched-baseline restore, the OFF drift control, the readiness gate, the
DV-symmetry scoping -- is preserved so the two runs are directly comparable, and the
sigma=1.0 sweep arm is an internal REPLICATION of 894's FAIL at the baseline threshold.

WHY A RETEST, AND THE HYPOTHESIS UNDER TEST (from the 894 autopsy).
894 ran green (18/18 readiness preconditions), non-degenerate, matched-baseline. Its
result was FAIL/weakens: the attribution-gate half (C1 attribution_selectivity, C2
context_differentiated_addressing -- the Moita 2004 dissociation) failed on 2/3 seeds;
the partiality half (C3) held on 2/3. But the autopsy found a concrete, testable
confound: across the three 894 seeds, FIRE-FRACTION and ATTRIBUTION-SELECTIVITY are
INVERSELY correlated. Measured on 894:

    seed  fire_frac(on)  attr_mass_excess  C1     C2     C3     C4
    45    0.4115  (LOW)  0.21314  (HIGH)   PASS   PASS   PASS   FAIL
    42    0.5912         0.00260           FAIL   FAIL   FAIL   FAIL
    43    0.9072  (HIGH) 0.03834  (LOW)    FAIL   FAIL   PASS   FAIL

The one seed with a clean positive attribution signal (45) has the LOWEST fire fraction;
the two near-chance seeds (42, 43) have the two HIGHEST. C4 (fire_frac <= 0.25, the
normal-theory ~1-SD sparsity ceiling) fails on ALL THREE seeds. This is the signature of
a gate that is firing far too often -- pooling ordinary, non-spike ticks (on which the
slot-attention softmax is near-uniform, entropy ~1.0) into the C1/C2 statistics and
diluting a genuine signal to chance -- rather than of an attribution mechanism that is
architecturally absent.

ROOT CAUSE OF THE OVER-FIRING (reconstructed from 894's recorded PE data, 2026-08-08).
The gate fires when the EMA-normalised PE z-score exceeds sigma
(``ree_core/amygdala/bla.py:483-497``): pe_z = (pe - running_mean) / running_std, with
the running mean/std an EMA at ``remap_pe_ema_alpha = 0.02`` (very slow). The driver
calls ``agent.bla.reset()`` at the START of every episode (see ``_run_phase``), which
re-initialises the EMA to mean=0.0 / std=0.1 (``remap_pe_std_init``). But the actual
harm-PE magnitude in this env is ~0.25-0.32 -- an order of magnitude above the mean's
0.0 init -- and at alpha=0.02 the running mean does NOT catch up to it within a 60-step
episode. So for most ticks of most episodes, pe_z = (0.28 - small_mean)/small_std is
comfortably > 1.0, and the gate fires. Replaying 894's own recorded per-tick PE
magnitudes through this exact EMA (with the per-episode reset) reproduces the measured
fire fractions closely (seed 43 recon 0.92 vs measured 0.91). Crucially, the three seeds
need DIFFERENT sigmas to reach the same fire fraction, because they accumulate different
numbers of BLA ticks per episode (570 / 194 / 853) and so warm the EMA to different
degrees -- which is precisely why a single higher sigma cannot land all seeds and a
SWEEP is the correct instrument.

  Faithfulness to the claim. MECH-074d's functional_restatement says the gate fires when
  PE "exceeds ~1 standard deviation of the RUNNING harm-PE distribution". The
  per-episode-reset EMA MIS-ESTIMATES that running SD (its mean is pinned near 0 while
  the true PE mean is ~0.28), so a NOMINAL config sigma of 1.0 corresponds to a
  much-GREATER-than-1-true-SD excursion being routine. Raising the nominal sigma
  compensates: a nominal 2.0-2.5 against the mis-estimated running std may correspond to
  a true ~1-SD excursion of the actual PE distribution -- i.e. the firing sparsity the
  claim's mechanism actually intends. So this sweep tests the claim's mechanism at the
  firing regime it was specified for, not a departure from it. (The deeper fix -- not
  resetting the PE EMA per episode, or setting ``remap_pe_std_init`` to the env's PE
  scale, or raising the SD-035 DEFAULT sigma -- is a SUBSTRATE change to bla.py and is
  OUT OF SCOPE for this threshold-recalibration retest; recorded as follow-on if the
  sweep confirms the over-firing hypothesis.)

WHY A REPRESENTATION DV AND NOT A BEHAVIOURAL ONE (unchanged from 894).
MECH-074d's own falsifier is stated at the representation level:

  "EXQ-B must show remap_signal fires on synthetic PE spikes above threshold AND
   preferentially perturbs attributed codes (partial remap, not wholesale); if remap
   fires on sub-threshold PE or perturbs untagged codes uniformly, the attribution
   gate is broken."                                  -- claims.yaml MECH-074d

So the DV is the CONTENT OF THE STORE THE REMAP WRITES TO --
``agent.e1.context_memory.memory`` -- plus the composition of the target sets the
attribution gate selects. Nothing downstream of the store is read, so no wall
(survival, recall performance, policy entropy) can gate the result.

SUBSTRATE WIRING VERIFIED (894 probe, 2026-08-08; unchanged -- same substrate).
  ree_core/amygdala/bla.py:472-531   remap gate: PE z-score > sigma AND non-empty
                                     attribution candidates; selects the top
                                     remap_code_fraction of candidates by |contribution|
  ree_core/agent.py:4450-4453        candidates := _get_context_memory_code_contributions
                                     (slot-attention softmax over ContextMemory)
  ree_core/agent.py:4488-4493        fire -> _apply_bla_context_remap
  ree_core/agent.py:3889-3913        the write: mem[i] <- (1-b)*mem[i] + b*write_signal
                                     for each targeted slot i
  ree_core/predictors/e1_deep.py:179 ContextMemory is constructed unconditionally

  ``use_e2_harm_a=True`` IS REQUIRED (894 probe): without it ``_harm_a_pred_prev`` is
  None and remap_signal can NEVER fire (0 fires in 296 ticks). Set identically in ALL
  arms. The sweep changes ONLY ``bla_remap_pe_sigma_threshold`` between ON arms; every
  other knob (arousal, code fraction, blend, EMA alpha, std init) is held at the 894
  values so the sweep isolates the threshold.

DESIGN -- 5 arms x 3 distinct seeds [42, 43, 45]. (Seed 44 excluded per the standing
per-seed early-episode-death instability on this reef-config env family -- EXQ-539-540,
V3-EXQ-538a. 45 substituted. Same seeds as 894 for direct comparability.)

  ARM_REMAP_OFF   bla_remap_pe_sigma_threshold = 1e9   (gate can never open; C3 drift
                                                        control, one per seed)
  ARM_SIGMA_10    sigma = 1.0   (SD-035 default -- REPLICATES 894's over-firing FAIL)
  ARM_SIGMA_15    sigma = 1.5
  ARM_SIGMA_20    sigma = 2.0
  ARM_SIGMA_25    sigma = 2.5

  The four ON arms are a monotone dose-response in sigma. Reconstruction from 894's PE
  data predicts fire fraction falls from ~0.5-0.9 (sigma 1.0) toward the ~0.16-0.25
  target band by sigma 2.0-2.5, seed-dependent -- so the sweep spans the over-firing
  regime through the intended-sparsity regime. A high-sigma arm that sparsifies BELOW
  the readiness floor (< MIN_REMAP_EVENTS fires, or a context under-sampled) goes gate-
  RED for that (seed, arm) cell and is SCOPED OUT of scoring there -- it does not vacate
  any other arm or seed (the V3-EXQ-785 rule, via precondition_gate).

  REMAP IS DISABLED IN ALL ARMS DURING P0, AND THE MEASUREMENT WINDOW IS A MATCHED-
  BASELINE REPEATED-MEASURES DESIGN (both load-bearing, both inherited from 894):
  (a) The remap writes IN PLACE to the store the DV reads, so it is held shut during P0
      (P0_REMAP_SIGMA=1e9 in every arm); each arm's threshold is installed on
      ``agent.bla.config`` only at the P0/P1 boundary, so all arms leave P0 from a
      matched, differentiated store.
  (b) With the gate open the remap can homogenise the store WITHIN the first measurement
      episode, after which the attention softmax is uniform BY CONSTRUCTION and pooling
      those fires would alias "gate vacuous" with "gate destroyed its own substrate". So
      the measurement window RESTORES ContextMemory to its end-of-P0 snapshot at the
      START of every measurement episode, identically in EVERY arm. Each episode is then
      an independent replicate from the same differentiated baseline; C1/C2 are always
      evaluated on a non-degenerate store, and C3 is a clean within-episode ON-vs-matched-
      OFF contrast. The restore touches ONLY ContextMemory content (encoder/E2/BLA carry
      through) and is applied to all arms, so ordinary-write drift stays matched and C3's
      ON/OFF ratio still divides it out.

  ARM_REMAP_OFF is NOT decoration: ordinary ContextMemory writes move slots on their
  own, so "targeted slots moved" is not by itself evidence of anything. OFF is the
  matched drift control that C3 divides by, shared across all four ON sigma arms at the
  same seed.

DV-SYMMETRY DECLARATION (one per arm, per the V3-EXQ-604c rule).
  The DV group is {per-slot content of cm.memory, summarised as per-slot displacement
  and as the spread of off-diagonal pairwise cosine similarity} plus {the index
  composition of the remap target sets}. Symmetries: a permutation of slot indices, and
  any transform applied identically to every slot.
  EACH ON arm (ARM_SIGMA_10/15/20/25) -- the manipulation is enabling an in-place,
    PER-SLOT blend of a SELECTED MINORITY of slots toward a common write_signal. It is
    neither a slot permutation nor a uniform-across-all transform, so it is NOT invariant
    under the DV's symmetry group; a blend of a strict subset toward a common vector
    strictly contracts pairwise distances among the written slots and MUST move the
    spread statistic. Confirmed live in the 894 probe (5 of 16 slots displaced per event).
    Raising sigma changes only HOW OFTEN the manipulation fires, not its symmetry class,
    so every ON arm remains non-invariant under the DV group.
  ARM_REMAP_OFF -- the manipulation is the ABSENCE of that write (threshold 1e9). The DV
    registers presence-vs-absence directly.
  C1/C2/C4 are measured only in the ON arms and are SCOPED OUT of ARM_REMAP_OFF via
  applies_to/structural_max rather than failed by it (the V3-EXQ-785 rule): with the gate
  shut there are no fires and no target sets, so those statistics are not meaningful there
  -- and that scoping must NOT vacate any ON arm.

CRITERIA (all thresholds are constants below, pre-registered, never derived post-hoc;
identical to 894 -- only the ARM they are evaluated on changes).
  C1 attribution_selectivity  [LOAD-BEARING] mean over fire ticks of (mass on the k
      selected candidates) minus the k/n chance floor > ATTR_MASS_EXCESS_MARGIN.
  C2 context_differentiated_addressing [LOAD-BEARING] mean WITHIN-context target-set
      Jaccard minus mean CROSS-context Jaccard > CONTEXT_JACCARD_GAP_MARGIN.
  C3 partial_not_wholesale    [LOAD-BEARING] ON-arm mean end-of-EPISODE slot
      differentiation >= SLOT_DIFF_RATIO_FLOOR x the matched OFF-arm mean at the same
      seed, both from the same restored baseline store.
  C4 pe_spike_sparsity        fraction of BLA ticks carrying a fire <= FIRE_FRAC_CEIL.

  COMBINATION RULE (recorded explicitly in the manifest, per the V3-EXQ-846 rule --
  this is an OR over arms, NOT a plain AND, so the per-arm criterion booleans alone
  under-read the result):
    - For EACH ON sigma arm: a criterion is MET when it holds on >= SEEDS_PASS_MIN of
      that arm's GREEN (gate-passing) seeds; the arm PASSES iff C1 AND C2 AND C3 AND C4.
    - RUN PASS iff ANY ON sigma arm passes -- i.e. the sweep finds an operating point at
      which the attribution gate is selective, context-differentiated, partial and
      sparse on >= 2/3 seeds. This is the "recalibrated threshold recovers the signal"
      outcome.
  DIRECTION:
    - supports : some ON sigma arm fully passes (recalibration recovers the mechanism ->
      894's FAIL was a calibration artifact, not architectural absence).
    - mixed    : no arm fully passes, BUT selectivity RECOVERS with sigma -- the
      attribution-gate half (C1 AND C2) is met at some higher sigma, OR mean attribution
      selectivity rises monotonically as fire-fraction falls across the sweep (Spearman
      of mean mass_excess vs sigma > 0 with the top sigma clearing the C1 margin on >= 2
      green seeds). The over-firing/dilution hypothesis is directionally confirmed even
      if a full 4-criteria pass is not reached.
    - weakens  : selectivity does NOT recover at any sigma (flat / uncorrelated with the
      sweep) -- the attribution gate is genuinely vacuous, and 894's FAIL stands as a
      real weakening rather than a calibration artifact.

READINESS vs VERDICT (unchanged from 894). Readiness asserts the mechanism's INPUTS are
non-degenerate (differentiated slots at window start; PE actually varies; enough fires;
both contexts sampled); the criteria then judge the mechanism. Readiness is measured on
positive controls, never on the statistic a criterion routes on.

SELF-ROUTE. If NO ON arm has any green (seed, arm) cell the run reports
non_degenerate=false and labels ``substrate_not_ready_requeue`` -- never a substrate
verdict.

experiment_purpose=evidence. claim_ids: MECH-074d only. Wall-independent: every readout
is internal to the ContextMemory store and the BLA gate; no behavioural outcome is read.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from experiment_protocol import emit_outcome  # noqa: E402
from _harness import StepHarness  # noqa: E402
from _lib.arm_fingerprint import arm_cell  # noqa: E402
from _lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from _lib.precondition_gate import (  # noqa: E402
    PreconditionSpec,
    aggregate_arm_gates,
    arm_criteria_non_degenerate,
    assert_no_structurally_unsatisfiable_gate,
    evaluate_arm_gate,
)
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_894a_mech074d_bla_remap_attribution_selectivity"
QUEUE_ID = "V3-EXQ-894a"
SUPERSEDES = "v3_exq_894_mech074d_bla_remap_attribution_selectivity_20260808T005219Z_v3"
CLAIM_IDS: List[str] = ["MECH-074d"]
EXPERIMENT_PURPOSE = "evidence"

# Seed 44 excluded (recurring early-episode-death instability on this reef-config env
# family: EXQ-539-540, V3-EXQ-538a). 45 substituted.
SEEDS = [42, 43, 45]

# Encoder dims -- matched to the sibling MECH-074 probe V3-EXQ-888 so the BLA arousal
# calibration and the env kwargs carry over unchanged.
WORLD_DIM = 32
SELF_DIM = 32
HARM_DIM = 32       # z_harm   (SD-010 sensory)
HARM_A_DIM = 16     # z_harm_a (SD-011 affective) -- the stream the PE is taken on
HARM_HISTORY_LEN = 10

P0_EPISODES = 20        # encoder/world warmup, remap gate held SHUT in both arms
MEASURE_EPISODES = 16    # frozen-encoder measurement window, per-arm gate installed
STEPS_PER_EPISODE = 60
TOTAL_EPISODES = P0_EPISODES + MEASURE_EPISODES   # = episodes_per_run in the queue

# BLA calibration -- identical across arms; ONLY the PE sigma threshold differs.
BLA_AROUSAL_THRESHOLD_ON = 0.4
BLA_AROUSAL_PEAK = 0.7
BLA_WINDOW_STEPS = 5
BLA_REMAP_CODE_FRACTION = 0.33     # SD-035 default (Moita 2004 ~30-35% overwrite)
BLA_CONTEXT_REMAP_BLEND = 0.5      # SD-035 default
# The sigma SWEEP -- the ONE dimension 894a changes. 1.0 is the SD-035 default and
# replicates 894's over-firing FAIL; 1.5/2.0/2.5 sparsify the gate toward the
# normal-theory ~1-SD firing regime the claim's mechanism intends (see the docstring's
# ROOT CAUSE section for the EMA-normalisation reconstruction that motivates the range).
REMAP_SIGMA_SWEEP = [1.0, 1.5, 2.0, 2.5]
REMAP_SIGMA_OFF = 1.0e9            # gate can never open (drift control)
P0_REMAP_SIGMA = 1.0e9             # all arms, during P0 only

# ---- Pre-registered acceptance thresholds (constants; never derived post-hoc) ----
# Readiness floors.
MIN_REMAP_EVENTS = 20         # ON arm: enough fires to estimate the target statistics
SLOT_DIFF_STD_FLOOR = 0.02    # both arms: slots differentiated at window START
PE_SPREAD_FLOOR = 1.0e-6      # both arms: the PE the gate reads must actually vary
MIN_WINDOWS_PER_CONTEXT = 4   # ON arm: both contexts sampled enough for C2

# Verdict thresholds.
# C1: chance mass on k of n candidates is k/n. 0.05 is a 16-percentage-point-relative
# excess on the k/n=0.3125 floor these dims give -- comfortably above the ~0.0004
# the probe measured, and far below what any genuinely selective gate would produce.
ATTR_MASS_EXCESS_MARGIN = 0.05
# C2: a constant target set scores exactly 0.0 on this gap by construction.
CONTEXT_JACCARD_GAP_MARGIN = 0.05
# C3: the ON store must retain at least half the matched control's differentiation.
SLOT_DIFF_RATIO_FLOOR = 0.5
# C4: a > 1-SD excursion of a roughly symmetric PE distribution is ~0.16 of ticks;
# 0.25 is that with generous headroom, pre-registered from normal theory rather than
# tuned to the probe's ~0.5.
FIRE_FRAC_CEIL = 0.25
SEEDS_PASS_MIN = 2            # >= 2/3 seeds

ARM_OFF = "ARM_REMAP_OFF"


def _sigma_arm_id(sigma: float) -> str:
    """ARM_SIGMA_10 for 1.0, ARM_SIGMA_25 for 2.5 -- a stable, sortable label per sweep
    point (10 * sigma, zero-padded), used to namespace preconditions and fingerprints."""
    return f"ARM_SIGMA_{int(round(sigma * 10)):02d}"


# One OFF drift control + one ON arm per swept sigma. OFF FIRST so its per-seed
# differentiation baseline (the C3 denominator) is available when the ON arms evaluate.
ARMS: List[Dict[str, Any]] = [
    {"arm_id": ARM_OFF, "remap_sigma": REMAP_SIGMA_OFF, "remap_live": False},
] + [
    {"arm_id": _sigma_arm_id(s), "remap_sigma": float(s), "remap_live": True}
    for s in REMAP_SIGMA_SWEEP
]
# The ON arm ids, in sweep order, for the dose-response readout.
ON_ARM_IDS: List[str] = [_sigma_arm_id(s) for s in REMAP_SIGMA_SWEEP]

# Threat context: SD-022 scheduled limb-damage injection drives RELIABLE,
# policy-independent body damage, so ||z_harm_a|| moves for reasons that do not depend
# on the (frozen, untrained) policy finding hazards.
THREAT_ENV_KWARGS: Dict[str, Any] = dict(
    size=10,
    num_hazards=4,
    num_resources=4,
    hazard_harm=0.05,
    env_drift_interval=5,
    env_drift_prob=0.1,
    proximity_harm_scale=0.1,
    proximity_benefit_scale=0.05,
    proximity_approach_threshold=0.2,
    hazard_field_decay=0.5,
    resource_respawn_on_consume=True,
    use_proxy_fields=True,
    toroidal=False,
    harm_history_len=HARM_HISTORY_LEN,
    limb_damage_enabled=True,
    damage_increment=0.15,
    failure_prob_scale=0.3,
    heal_rate=0.002,
    scheduled_limb_damage_enabled=True,
    scheduled_limb_damage_interval=8,
    scheduled_limb_damage_prob=1.0,
    scheduled_limb_damage_magnitude=0.25,
    scheduled_limb_damage_limb_selection="all",
)
# Neutral context: no hazards AND no scheduled injection. Same obs dims, so one agent
# runs in both. The threat/neutral alternation is what gives C2 two predictive
# contexts to dissociate between.
NEUTRAL_ENV_KWARGS: Dict[str, Any] = dict(
    THREAT_ENV_KWARGS, num_hazards=0, scheduled_limb_damage_enabled=False,
)

CTX_THREAT = "threat"
CTX_NEUTRAL = "neutral"


# --------------------------------------------------------------------------------
# Readiness preconditions. Regime-conditioned with applies_to: the fire-dependent ones
# are structurally impossible in ARM_REMAP_OFF (the gate is shut BY DESIGN there, which
# is the manipulation, not a substrate failure), and ANDing them whole-run would vacate
# the ON arm this design exists to measure -- the V3-EXQ-785 defect.
# --------------------------------------------------------------------------------
PRECONDITION_SPECS: List[PreconditionSpec] = [
    PreconditionSpec(
        name="slot_differentiation_at_window_start",
        description=(
            "Std of the off-diagonal pairwise cosine similarity of "
            "ContextMemory.memory in the BASELINE store every measurement episode is "
            "restored to. If the slots were identical the attention softmax would be "
            "uniform BY CONSTRUCTION, so C1 would be starved rather than falsified. "
            "Measured after P0 with the remap gate held shut in BOTH arms, so both "
            "arms measure from the same differentiated baseline."
        ),
        control="post-P0 agent in the threat context, remap disabled throughout P0",
        threshold=SLOT_DIFF_STD_FLOOR,
        direction="lower",
        kind="readiness",
    ),
    PreconditionSpec(
        name="pe_distribution_spread",
        description=(
            "Std of the harm-PE magnitude the BLA gate reads across measurement ticks. "
            "A constant PE makes the sigma gate meaningless in either direction."
        ),
        control="threat/neutral alternation with scheduled limb-damage injection",
        threshold=PE_SPREAD_FLOOR,
        direction="lower",
        kind="readiness",
    ),
    PreconditionSpec(
        name="remap_events_sufficient",
        description=(
            "Number of remap fires in the measurement window. C1/C2 are statistics OVER "
            "fires; with none there is nothing to measure."
        ),
        control="bla_remap_pe_sigma_threshold=1.0 with use_e2_harm_a=True supplying "
                "z_harm_a_pred (verified in the 2026-08-08 probe: 72 fires / 137 ticks)",
        # A floor in this module is STRICT (met iff measured > threshold), so the
        # threshold is stated as N-1 to make "at least MIN_REMAP_EVENTS" mean exactly
        # that. Same idiom as V3-EXQ-888's n_traces_sufficient.
        threshold=float(MIN_REMAP_EVENTS - 1),
        direction="lower",
        kind="readiness",
        applies_to=lambda ctx: bool(ctx["remap_live"]),
        applies_note=(
            "not meaningful with the gate shut (sigma=1e9): zero fires is the "
            "manipulation in this arm, not a substrate failure"
        ),
        structural_max=lambda ctx: (0.0 if not ctx["remap_live"] else None),
    ),
    PreconditionSpec(
        name="both_contexts_fired",
        description=(
            "Minimum number of fire-bearing episodes in EACH of the threat and neutral "
            "contexts. C2 is a within-minus-cross-context contrast and is undefined if "
            "one context contributed no target sets."
        ),
        control="episodes alternate threat/neutral through the measurement window",
        # Strict floor -- see remap_events_sufficient. N-1 makes "at least
        # MIN_WINDOWS_PER_CONTEXT fire-bearing episodes in EACH context" mean that.
        threshold=float(MIN_WINDOWS_PER_CONTEXT - 1),
        direction="lower",
        kind="readiness",
        applies_to=lambda ctx: bool(ctx["remap_live"]),
        applies_note=(
            "not meaningful with the gate shut: no arm-OFF episode can carry a target "
            "set, by design"
        ),
        structural_max=lambda ctx: (0.0 if not ctx["remap_live"] else None),
    ),
]


# z_goal liveness recording. A fresh agent is built inside each cell, so the
# accumulator is used rather than holding every arm x seed agent alive to the end.
_ZG = ZGoalStreamAccumulator()


def _arm_ctx(arm: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "arm_id": arm["arm_id"],
        "remap_sigma": float(arm["remap_sigma"]),
        "remap_live": bool(arm["remap_live"]),
    }


def _make_env(seed: int, kwargs: Dict[str, Any]) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **kwargs)


def _build_config(env: CausalGridWorldV2) -> REEConfig:
    """Config is IDENTICAL across arms. The only arm difference is the PE sigma
    threshold, which is installed on agent.bla.config at the P0/P1 boundary so that
    P0 is bit-comparable across arms."""
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        alpha_world=0.9,
        world_dim=WORLD_DIM,
        self_dim=SELF_DIM,
        use_harm_stream=True,
        z_harm_dim=HARM_DIM,
        use_affective_harm_stream=True,
        z_harm_a_dim=HARM_A_DIM,
        harm_history_len=HARM_HISTORY_LEN,
        limb_damage_enabled=True,
        damage_increment=float(THREAT_ENV_KWARGS["damage_increment"]),
        failure_prob_scale=float(THREAT_ENV_KWARGS["failure_prob_scale"]),
        heal_rate=float(THREAT_ENV_KWARGS["heal_rate"]),
        # ARC-033 / ARC-058 harm-a forward model. LOAD-BEARING: without it
        # _harm_a_pred_prev stays None and remap_signal can never fire at all
        # (probe 2026-08-08: 0 fires in 296 ticks without it).
        use_e2_harm_a=True,
        # SD-035 amygdala analogue -- BLA on; CeA off (not under test).
        use_amygdala_analog=True,
        use_bla_analog=True,
        use_cea_analog=False,
        bla_arousal_threshold_on=BLA_AROUSAL_THRESHOLD_ON,
        bla_arousal_peak=BLA_AROUSAL_PEAK,
        bla_window_steps=BLA_WINDOW_STEPS,
        bla_remap_pe_sigma_threshold=P0_REMAP_SIGMA,   # shut during P0 in BOTH arms
        bla_remap_code_fraction=BLA_REMAP_CODE_FRACTION,
        bla_remap_requires_attribution=True,
        bla_context_remap_blend=BLA_CONTEXT_REMAP_BLEND,
        replay_diversity_enabled=True,
    )
    cfg.residue.valence_enabled = True
    return cfg


def _slot_diff_std(memory: torch.Tensor) -> float:
    """Spread of the off-diagonal pairwise cosine similarity of the slot matrix.

    This is the DV's differentiation statistic: 0 means every slot points the same way
    (a homogenised store), larger means the slots carry distinguishable content.
    """
    with torch.no_grad():
        m = torch.nn.functional.normalize(memory.detach().to(torch.float32), dim=-1)
        sim = m @ m.t()
        n = sim.shape[0]
        if n < 2:
            return 0.0
        off = sim[~torch.eye(n, dtype=torch.bool, device=sim.device)]
        return float(off.std().item())


def _slot_mean_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean per-slot cosine similarity between two snapshots of the store."""
    with torch.no_grad():
        x = torch.nn.functional.normalize(a.detach().to(torch.float32), dim=-1)
        y = torch.nn.functional.normalize(b.detach().to(torch.float32), dim=-1)
        return float((x * y).sum(dim=-1).mean().item())


def _jaccard(a: Sequence[int], b: Sequence[int]) -> float:
    sa, sb = set(int(x) for x in a), set(int(x) for x in b)
    union = sa | sb
    if not union:
        return 0.0
    return float(len(sa & sb) / len(union))


def _chance_jaccard(k: int, n: int) -> float:
    """Analytic expected Jaccard of two independent uniform k-subsets of n.

    E|A n B| ~ k^2/n, |A u B| = 2k - |A n B|. Recorded alongside the observed values
    so the C2 gap is interpretable, though C2 itself routes on the within-minus-cross
    contrast and does NOT need this floor.
    """
    if k <= 0 or n <= 0:
        return 0.0
    inter = (k * k) / float(n)
    union = (2.0 * k) - inter
    return float(inter / union) if union > 1e-9 else 0.0


def _run_phase(
    agent: REEAgent,
    threat_env: CausalGridWorldV2,
    neutral_env: CausalGridWorldV2,
    *,
    num_episodes: int,
    steps_per_episode: int,
    seed: int,
    episode_offset: int,
    total_episodes: int,
    train_mode: bool,
    record: bool,
    label: str,
    restore_base: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """Run alternating threat/neutral episodes.

    When ``restore_base`` is supplied, ContextMemory is restored to that snapshot at the
    START of every episode -- identically in both arms. That makes each episode an
    independent replicate measured from the same differentiated baseline, which is what
    keeps C1/C2 evaluable on a non-degenerate store even though the ON arm's own remap
    homogenises the store within a single episode (see the module docstring, (b)).

    When record=True, captures per-tick: the exact attribution-contribution vector the
    BLA gate consumed that tick, the remap target set it produced, the harm-PE
    magnitude, and the episode's context label; plus a per-episode snapshot of the
    ContextMemory store.

    The contributions are captured by wrapping ``_get_context_memory_code_contributions``
    so the recorded vector is the one the gate ACTUALLY saw, computed before the remap
    mutated the store on that same tick -- recomputing it after the step would read a
    store the remap had already changed.
    """
    cm = agent.e1.context_memory
    n_slots = int(cm.memory.shape[0])

    captured: Dict[str, Any] = {"contrib": None}
    orig_contrib = agent._get_context_memory_code_contributions

    def _contrib_spy(z_self, z_world):
        out = orig_contrib(z_self, z_world)
        captured["contrib"] = out
        return out

    if record:
        agent._get_context_memory_code_contributions = _contrib_spy

    fire_ticks: List[Dict[str, Any]] = []
    pe_magnitudes: List[float] = []
    n_bla_ticks = 0
    n_no_bla_tick = 0
    episodes_with_fire = {CTX_THREAT: 0, CTX_NEUTRAL: 0}
    per_episode: List[Dict[str, Any]] = []

    if train_mode:
        agent.train()
    else:
        agent.eval()

    try:
        for ep in range(num_episodes):
            is_threat = (ep % 2 == 0)
            ctx = CTX_THREAT if is_threat else CTX_NEUTRAL
            env = threat_env if is_threat else neutral_env
            harness = StepHarness(agent, env, train_mode=train_mode,
                                  seed=seed + episode_offset + ep)
            _, obs_dict = env.reset()
            agent.reset()
            if agent.bla is not None:
                agent.bla.reset()
            harness.reset()

            if restore_base is not None:
                with torch.no_grad():
                    cm.memory.data.copy_(restore_base)

            mem_before = cm.memory.data.detach().clone() if record else None
            ep_fires = 0
            ep_target_sets: List[List[int]] = []

            for _ in range(steps_per_episode):
                # Clear the BLA output latch IMMEDIATELY before the step so a tick on
                # which the BLA did not run is counted as such rather than re-recording
                # the previous tick's remap_signal as a fresh observation.
                agent._bla_last_output = None
                captured["contrib"] = None

                result = harness.step(obs_dict)

                if record:
                    out = agent._bla_last_output
                    if out is None:
                        n_no_bla_tick += 1
                    else:
                        n_bla_ticks += 1
                        pe_magnitudes.append(float(out.pe_magnitude))
                        sig = out.remap_signal or {}
                        if sig:
                            targets = sorted(int(k) for k in sig.keys())
                            contrib = captured["contrib"] or {}
                            vals = np.array(
                                [float(contrib.get(i, 0.0)) for i in range(n_slots)],
                                dtype=np.float64,
                            )
                            total = float(vals.sum())
                            k = len(targets)
                            sel_mass = (
                                float(vals[targets].sum() / total) if total > 1e-12
                                else 0.0
                            )
                            order = np.sort(vals)[::-1]
                            norm_entropy = 0.0
                            if total > 1e-12 and n_slots > 1:
                                p = vals / total
                                nz = p[p > 0]
                                norm_entropy = float(
                                    -(nz * np.log(nz)).sum() / math.log(n_slots)
                                )
                            fire_ticks.append({
                                "context": ctx,
                                "targets": targets,
                                "k": int(k),
                                "n_candidates": int(n_slots),
                                "selected_mass": sel_mass,
                                "chance_mass": float(k) / float(n_slots),
                                "mass_excess": sel_mass - (float(k) / float(n_slots)),
                                "attr_norm_entropy": norm_entropy,
                                "attr_top_value": float(order[0]) if order.size else 0.0,
                                "pe_magnitude": float(out.pe_magnitude),
                                "pe_baseline_std": float(out.pe_baseline_std),
                            })
                            ep_fires += 1
                            ep_target_sets.append(targets)

                obs_dict = result.next_obs_dict
                if result.done:
                    break

            if record:
                mem_after = cm.memory.data.detach().clone()
                disp = (mem_after - mem_before).norm(dim=-1)
                tgt_counts = np.zeros(n_slots, dtype=np.float64)
                for ts in ep_target_sets:
                    for i in ts:
                        tgt_counts[i] += 1.0
                touched = tgt_counts > 0
                d = disp.detach().cpu().numpy().astype(np.float64)
                per_episode.append({
                    "episode": int(ep),
                    "context": ctx,
                    "n_fires": int(ep_fires),
                    "slot_diff_std_start": _slot_diff_std(mem_before),
                    "slot_diff_std_end": _slot_diff_std(mem_after),
                    "slot_retention_cos": _slot_mean_cos(mem_before, mem_after),
                    "mean_disp_targeted": (
                        float(d[touched].mean()) if bool(touched.any()) else 0.0),
                    "mean_disp_untargeted": (
                        float(d[~touched].mean()) if bool((~touched).any()) else 0.0),
                    "n_slots_targeted": int(touched.sum()),
                })
                if ep_fires > 0:
                    episodes_with_fire[ctx] += 1

            done_ep = episode_offset + ep + 1
            if done_ep % 5 == 0 or done_ep == total_episodes:
                print(f"  [train] {label} ep {done_ep}/{total_episodes}", flush=True)
    finally:
        if record:
            agent._get_context_memory_code_contributions = orig_contrib

    return {
        "fire_ticks": fire_ticks,
        "pe_magnitudes": pe_magnitudes,
        "n_bla_ticks": int(n_bla_ticks),
        "n_no_bla_tick": int(n_no_bla_tick),
        "episodes_with_fire": episodes_with_fire,
        "per_episode": per_episode,
    }


def _summarise(
    meas: Dict[str, Any],
    *,
    n_slots: int,
    slot_diff_baseline: float,
) -> Dict[str, Any]:
    fires = meas["fire_ticks"]
    pes = meas["pe_magnitudes"]
    n_bla = int(meas["n_bla_ticks"])
    eps = meas["per_episode"]

    # C3 routes on the MEAN end-of-EPISODE differentiation, because every episode is
    # restored to the same baseline store -- so this is a per-replicate statistic, not
    # a single end-of-window reading contaminated by the previous episodes' damage.
    diff_end_mean = float(np.mean([e["slot_diff_std_end"] for e in eps])) if eps else 0.0
    diff_start_mean = (
        float(np.mean([e["slot_diff_std_start"] for e in eps])) if eps else 0.0)
    retention_mean = (
        float(np.mean([e["slot_retention_cos"] for e in eps])) if eps else 0.0)

    out: Dict[str, Any] = {
        "n_bla_ticks": n_bla,
        "n_no_bla_tick": int(meas["n_no_bla_tick"]),
        "n_remap_events": len(fires),
        "fire_fraction": (len(fires) / n_bla) if n_bla else 0.0,
        "pe_magnitude_mean": float(np.mean(pes)) if pes else 0.0,
        "pe_magnitude_std": float(np.std(pes)) if pes else 0.0,
        "slot_diff_std_baseline": float(slot_diff_baseline),
        "slot_diff_std_episode_start_mean": diff_start_mean,
        "slot_diff_std_episode_end_mean": diff_end_mean,
        "slot_retention_cos_episode_mean": retention_mean,
        "n_measure_episodes": len(eps),
        "n_slots": int(n_slots),
        "episodes_with_fire_threat": int(meas["episodes_with_fire"][CTX_THREAT]),
        "episodes_with_fire_neutral": int(meas["episodes_with_fire"][CTX_NEUTRAL]),
        # C1 / C2 statistics -- 0.0 when there are no fires, which only ever happens in
        # the gate-shut arm where both are scoped out anyway.
        "attr_mass_excess_mean": 0.0,
        "attr_selected_mass_mean": 0.0,
        "attr_chance_mass": 0.0,
        "attr_norm_entropy_mean": 0.0,
        "target_set_k_mean": 0.0,
        "jaccard_within_context_mean": 0.0,
        "jaccard_cross_context_mean": 0.0,
        "jaccard_context_gap": 0.0,
        "jaccard_chance": 0.0,
        "n_jaccard_within_pairs": 0,
        "n_jaccard_cross_pairs": 0,
    }
    if not fires:
        return out

    out["attr_mass_excess_mean"] = float(np.mean([f["mass_excess"] for f in fires]))
    out["attr_selected_mass_mean"] = float(np.mean([f["selected_mass"] for f in fires]))
    out["attr_chance_mass"] = float(np.mean([f["chance_mass"] for f in fires]))
    out["attr_norm_entropy_mean"] = float(
        np.mean([f["attr_norm_entropy"] for f in fires]))
    ks = [int(f["k"]) for f in fires]
    out["target_set_k_mean"] = float(np.mean(ks))
    out["jaccard_chance"] = _chance_jaccard(int(round(float(np.mean(ks)))), n_slots)

    # C2: within-context vs cross-context target-set overlap. Pairs are capped so a
    # long window does not blow up into O(n^2) work; the cap is applied identically to
    # both pair classes so it cannot bias the gap.
    MAX_PAIRS = 4000
    by_ctx: Dict[str, List[List[int]]] = {CTX_THREAT: [], CTX_NEUTRAL: []}
    for f in fires:
        by_ctx[f["context"]].append(f["targets"])

    within: List[float] = []
    for ctx in (CTX_THREAT, CTX_NEUTRAL):
        sets = by_ctx[ctx]
        for a, b in itertools.islice(itertools.combinations(sets, 2), MAX_PAIRS):
            within.append(_jaccard(a, b))
    cross: List[float] = []
    for a, b in itertools.islice(
        itertools.product(by_ctx[CTX_THREAT], by_ctx[CTX_NEUTRAL]), MAX_PAIRS
    ):
        cross.append(_jaccard(a, b))

    out["n_jaccard_within_pairs"] = len(within)
    out["n_jaccard_cross_pairs"] = len(cross)
    if within:
        out["jaccard_within_context_mean"] = float(np.mean(within))
    if cross:
        out["jaccard_cross_context_mean"] = float(np.mean(cross))
    if within and cross:
        out["jaccard_context_gap"] = (
            out["jaccard_within_context_mean"] - out["jaccard_cross_context_mean"])
    return out


def _run_cell(seed: int, arm: Dict[str, Any], *, dry: bool) -> Dict[str, Any]:
    arm_id = arm["arm_id"]
    print(f"Seed {seed} Condition {arm_id}", flush=True)

    # Dry-run sizing: measure_episodes must stay >= 2 * MIN_WINDOWS_PER_CONTEXT so the
    # readiness gate is REACHABLE on the smoke. A smaller smoke would self-route
    # substrate_not_ready_requeue for a reason that is purely an artifact of the smoke's
    # own episode count, which proves nothing about the full grid.
    p0_eps = 3 if dry else P0_EPISODES
    meas_eps = (2 * MIN_WINDOWS_PER_CONTEXT) if dry else MEASURE_EPISODES
    steps = 15 if dry else STEPS_PER_EPISODE
    total_eps = p0_eps + meas_eps

    config_slice = {
        "arm_id": arm_id,
        "remap_sigma_measurement": float(arm["remap_sigma"]),
        "remap_sigma_p0": float(P0_REMAP_SIGMA),
        "bla_remap_code_fraction": BLA_REMAP_CODE_FRACTION,
        "bla_context_remap_blend": BLA_CONTEXT_REMAP_BLEND,
        "bla_remap_requires_attribution": True,
        "bla_arousal_threshold_on": BLA_AROUSAL_THRESHOLD_ON,
        "bla_arousal_peak": BLA_AROUSAL_PEAK,
        "bla_window_steps": BLA_WINDOW_STEPS,
        "use_e2_harm_a": True,
        "world_dim": WORLD_DIM,
        "self_dim": SELF_DIM,
        "harm_dim": HARM_DIM,
        "harm_a_dim": HARM_A_DIM,
        "p0_episodes": p0_eps,
        "measure_episodes": meas_eps,
        "steps_per_episode": steps,
    }
    label = f"mech074d seed={seed} arm={arm_id}"

    with arm_cell(
        seed,
        config_slice=config_slice,
        script_path=Path(__file__),
        config_slice_declared=True,
        # No extra ineligibility reason: each cell builds its OWN envs and agent, and
        # arm_cell resets all RNG on entry, so a cell is a pure function of
        # (substrate, config_slice, seed). The fingerprint is driver-INCLUSIVE (the
        # default), so a same-driver successor can reuse these cells; cross-driver
        # reuse would additionally need this arm's path factored into
        # experiments/_lib/baselines/, which is NOT done here and is recorded as
        # follow-on rather than claimed.
    ) as cell:
        threat_env = _make_env(seed, THREAT_ENV_KWARGS)
        neutral_env = _make_env(seed + 1, NEUTRAL_ENV_KWARGS)
        cfg = _build_config(threat_env)
        agent = REEAgent(cfg)
        cm = agent.e1.context_memory
        n_slots = int(cm.memory.shape[0])

        # ---- P0: warmup with the remap gate held SHUT in BOTH arms. ----
        _run_phase(
            agent, threat_env, neutral_env,
            num_episodes=p0_eps, steps_per_episode=steps, seed=seed,
            episode_offset=0, total_episodes=total_eps,
            train_mode=True, record=False, label=label,
        )

        # ---- Install the arm's gate, then measure with the encoder frozen. ----
        # The end-of-P0 store is the BASELINE every measurement episode is restored to,
        # in both arms identically (module docstring (b)).
        mem_baseline = cm.memory.data.detach().clone()
        slot_diff_baseline = _slot_diff_std(mem_baseline)
        if agent.bla is not None:
            agent.bla.config.remap_pe_sigma_threshold = float(arm["remap_sigma"])

        meas = _run_phase(
            agent, threat_env, neutral_env,
            num_episodes=meas_eps, steps_per_episode=steps, seed=seed,
            episode_offset=p0_eps, total_episodes=total_eps,
            train_mode=False, record=True, label=label,
            restore_base=mem_baseline,
        )
        _ZG.observe(agent)   # AFTER stepping -- reads the counters at call time

        summ = _summarise(
            meas, n_slots=n_slots, slot_diff_baseline=slot_diff_baseline,
        )

        row: Dict[str, Any] = {
            "seed": int(seed),
            "arm": arm_id,
            "remap_sigma": float(arm["remap_sigma"]),
            "remap_live": bool(arm["remap_live"]),
            **summ,
            # Generous recording: the full per-episode series, so a successor can
            # re-derive any other displacement/retention statistic without re-running.
            "per_episode": meas["per_episode"],
            # The per-fire records, capped so the manifest stays a sane size while
            # still carrying the raw distribution the criteria summarise.
            "fire_records_sample": meas["fire_ticks"][:400],
            "n_fire_records_total": len(meas["fire_ticks"]),
            "pe_magnitude_sample": [float(x) for x in meas["pe_magnitudes"][:400]],
        }
        cell.stamp(row)

    gate = evaluate_arm_gate(
        arm_id,
        _arm_ctx(arm),
        PRECONDITION_SPECS,
        measured={
            "slot_differentiation_at_window_start": float(
                row["slot_diff_std_baseline"]),
            "pe_distribution_spread": float(row["pe_magnitude_std"]),
            "remap_events_sufficient": float(row["n_remap_events"]),
            "both_contexts_fired": float(min(
                row["episodes_with_fire_threat"], row["episodes_with_fire_neutral"])),
        },
    )
    row["arm_gate"] = gate

    # Per-cell verdict (progress line only; the cohort evaluation is authoritative).
    if arm["remap_live"]:
        cell_pass = bool(
            gate["gate_green"]
            and row["attr_mass_excess_mean"] > ATTR_MASS_EXCESS_MARGIN
            and row["jaccard_context_gap"] > CONTEXT_JACCARD_GAP_MARGIN
            and row["fire_fraction"] <= FIRE_FRAC_CEIL
        )
    else:
        cell_pass = bool(gate["gate_green"] and row["n_remap_events"] == 0)
    print(f"verdict: {'PASS' if cell_pass else 'FAIL'}", flush=True)
    row["cell_pass"] = bool(cell_pass)
    return row


def _spearman_sign(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Spearman rank correlation of two equal-length sequences (no scipy dependency).

    Returns 0.0 for < 2 points or a constant vector. Used only to report the SIGN and
    rough strength of the dose-response (mean attribution selectivity vs sigma); it is a
    diagnostic, not a gate."""
    n = len(xs)
    if n < 2 or len(ys) != n:
        return 0.0
    xr = np.argsort(np.argsort(np.asarray(xs, dtype=np.float64))).astype(np.float64)
    yr = np.argsort(np.argsort(np.asarray(ys, dtype=np.float64))).astype(np.float64)
    xr -= xr.mean(); yr -= yr.mean()
    denom = float(np.sqrt((xr * xr).sum() * (yr * yr).sum()))
    return float((xr * yr).sum() / denom) if denom > 1e-12 else 0.0


def _evaluate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Sweep evaluation. For each ON sigma arm, tally C1-C4 over that arm's GREEN
    (gate-passing) seeds against the matched OFF drift control at the same seed; an arm
    PASSES iff all four are met on >= seeds_needed green seeds. The RUN passes iff ANY ON
    arm passes (an OR over the sweep, recorded explicitly in combination_rule). Direction
    keys on whether the ATTRIBUTION gate (C1+C2, the Moita 2004 dissociation) RECOVERS as
    the gate is sparsified -- the over-firing/dilution hypothesis 894a exists to test."""
    by_seed: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for r in rows:
        by_seed.setdefault(int(r["seed"]), {})[r["arm"]] = r

    n_seeds_run = len(by_seed)
    # Pre-registered 3-seed cohort -> SEEDS_PASS_MIN (2 of 3). Relaxed only when fewer
    # seeds ran than were pre-registered (a smoke), so an unreachable criterion does not
    # force a misleading verdict on a run that never had the seeds to test it.
    seeds_needed = min(SEEDS_PASS_MIN, n_seeds_run) if n_seeds_run else SEEDS_PASS_MIN

    per_arm_per_seed: List[Dict[str, Any]] = []   # one row per (ON arm, seed)
    per_arm: List[Dict[str, Any]] = []            # one summary per ON sigma arm
    for arm_id in ON_ARM_IDS:
        sigma = float(next(a["remap_sigma"] for a in ARMS if a["arm_id"] == arm_id))
        green_seeds: List[int] = []
        c1_ok = c2_ok = c3_ok = c4_ok = 0
        arm_mass_excess: List[float] = []
        arm_fire_frac: List[float] = []
        arm_jac_gap: List[float] = []
        arm_ratio: List[float] = []
        for seed, arms in sorted(by_seed.items()):
            on = arms.get(arm_id)
            off = arms.get(ARM_OFF)
            if on is None or off is None:
                continue
            gate_green = bool(on.get("arm_gate", {}).get("gate_green"))

            mass_excess = float(on["attr_mass_excess_mean"])
            jac_gap = float(on["jaccard_context_gap"])
            diff_on = float(on["slot_diff_std_episode_end_mean"])
            diff_off = float(off["slot_diff_std_episode_end_mean"])
            ratio = (diff_on / diff_off) if diff_off > 1e-12 else 0.0
            fire_frac = float(on["fire_fraction"])

            s_c1 = bool(mass_excess > ATTR_MASS_EXCESS_MARGIN)
            s_c2 = bool(jac_gap > CONTEXT_JACCARD_GAP_MARGIN)
            s_c3 = bool(ratio >= SLOT_DIFF_RATIO_FLOOR)
            s_c4 = bool(fire_frac <= FIRE_FRAC_CEIL)

            # Only GREEN (gate-passing) cells count toward an arm's criterion tally. A
            # red cell (e.g. high sigma sparsified fires below the readiness floor) is
            # scoped out of scoring for this arm+seed and does not vacate any other.
            if gate_green:
                green_seeds.append(seed)
                c1_ok += int(s_c1); c2_ok += int(s_c2)
                c3_ok += int(s_c3); c4_ok += int(s_c4)
                arm_mass_excess.append(mass_excess)
                arm_fire_frac.append(fire_frac)
                arm_jac_gap.append(jac_gap)
                arm_ratio.append(ratio)

            per_arm_per_seed.append({
                "arm": arm_id,
                "remap_sigma": sigma,
                "seed": seed,
                "gate_green": gate_green,
                "attr_mass_excess_on": mass_excess,
                "attr_selected_mass_on": float(on["attr_selected_mass_mean"]),
                "attr_chance_mass_on": float(on["attr_chance_mass"]),
                "attr_norm_entropy_on": float(on["attr_norm_entropy_mean"]),
                "jaccard_within_on": float(on["jaccard_within_context_mean"]),
                "jaccard_cross_on": float(on["jaccard_cross_context_mean"]),
                "jaccard_gap_on": jac_gap,
                "jaccard_chance_on": float(on["jaccard_chance"]),
                "slot_diff_std_episode_end_mean_on": diff_on,
                "slot_diff_std_episode_end_mean_off": diff_off,
                "slot_diff_ratio_on_over_off": ratio,
                "slot_diff_std_baseline_on": float(on["slot_diff_std_baseline"]),
                "slot_diff_std_baseline_off": float(off["slot_diff_std_baseline"]),
                "fire_fraction_on": fire_frac,
                "n_remap_events_on": int(on["n_remap_events"]),
                "n_remap_events_off": int(off["n_remap_events"]),
                "c1_attribution_selectivity": s_c1,
                "c2_context_differentiated_addressing": s_c2,
                "c3_partial_not_wholesale": s_c3,
                "c4_pe_spike_sparsity": s_c4,
                # A seed "passes" for this arm only if green AND all four criteria hold.
                "seed_pass": bool(gate_green and s_c1 and s_c2 and s_c3 and s_c4),
            })

        n_green = len(green_seeds)
        eligible = n_green >= seeds_needed
        c1_met = bool(eligible and c1_ok >= seeds_needed)
        c2_met = bool(eligible and c2_ok >= seeds_needed)
        c3_met = bool(eligible and c3_ok >= seeds_needed)
        c4_met = bool(eligible and c4_ok >= seeds_needed)
        arm_pass = bool(c1_met and c2_met and c3_met and c4_met)
        per_arm.append({
            "arm": arm_id,
            "remap_sigma": sigma,
            "n_green_seeds": n_green,
            "green_seeds": list(green_seeds),
            "c1_seeds_ok": c1_ok, "c2_seeds_ok": c2_ok,
            "c3_seeds_ok": c3_ok, "c4_seeds_ok": c4_ok,
            "c1_attribution_selectivity_met": c1_met,
            "c2_context_differentiated_addressing_met": c2_met,
            "c3_partial_not_wholesale_met": c3_met,
            "c4_pe_spike_sparsity_met": c4_met,
            "arm_pass": arm_pass,
            "attribution_gate_half_met": bool(c1_met and c2_met),
            "partiality_half_met": bool(c3_met and c4_met),
            "mean_attr_mass_excess": (
                float(np.mean(arm_mass_excess)) if arm_mass_excess else None),
            "mean_fire_fraction": (
                float(np.mean(arm_fire_frac)) if arm_fire_frac else None),
            "mean_jaccard_gap": (
                float(np.mean(arm_jac_gap)) if arm_jac_gap else None),
            "mean_slot_diff_ratio": (
                float(np.mean(arm_ratio)) if arm_ratio else None),
        })

    # ---- Dose-response over the swept sigmas (arms with at least one green seed). ----
    sweep = [a for a in per_arm if a["n_green_seeds"] >= 1]
    dr_sigmas = [a["remap_sigma"] for a in sweep]
    dr_mass = [a["mean_attr_mass_excess"] for a in sweep]
    dr_fire = [a["mean_fire_fraction"] for a in sweep]
    spearman_mass_vs_sigma = _spearman_sign(dr_sigmas, dr_mass) if len(sweep) >= 2 else 0.0
    spearman_fire_vs_sigma = _spearman_sign(dr_sigmas, dr_fire) if len(sweep) >= 2 else 0.0

    # Does the HIGHEST-sigma arm that has enough green seeds recover C1 (attribution
    # selectivity) on >= seeds_needed seeds, AND beat the lowest-sigma arm's mean
    # selectivity? That is the "attribution recovers with sparsification" signal.
    attribution_dose_recovers = False
    if len(sweep) >= 2:
        top = max(sweep, key=lambda a: a["remap_sigma"])
        bot = min(sweep, key=lambda a: a["remap_sigma"])
        top_c1_recovers = bool(
            top["n_green_seeds"] >= seeds_needed and top["c1_seeds_ok"] >= seeds_needed)
        rose = bool(
            top["mean_attr_mass_excess"] is not None
            and bot["mean_attr_mass_excess"] is not None
            and top["mean_attr_mass_excess"] > bot["mean_attr_mass_excess"])
        attribution_dose_recovers = bool(top_c1_recovers and rose)

    any_gate_half = any(a["attribution_gate_half_met"] for a in per_arm)
    any_partial_half = any(a["partiality_half_met"] for a in per_arm)
    passing_arms = [a for a in per_arm if a["arm_pass"]]
    outcome_pass = bool(passing_arms)
    best_arm = (
        min(passing_arms, key=lambda a: a["remap_sigma"]) if passing_arms
        else (max(sweep, key=lambda a: (a["c1_seeds_ok"], -a["remap_sigma"]))
              if sweep else None))

    # Any-arm criterion booleans (for the top-level criteria[] list + summary). An OR
    # over arms -- the per-arm booleans in per_arm[] carry the sigma-resolved detail.
    c1_met_any = any(a["c1_attribution_selectivity_met"] for a in per_arm)
    c2_met_any = any(a["c2_context_differentiated_addressing_met"] for a in per_arm)
    c3_met_any = any(a["c3_partial_not_wholesale_met"] for a in per_arm)
    c4_met_any = any(a["c4_pe_spike_sparsity_met"] for a in per_arm)

    # ---- Direction. supports = some sigma fully passes; mixed = attribution gate
    # RECOVERS with sparsification (some sigma meets C1 AND C2, or a monotone dose-
    # response in selectivity with the top sigma clearing C1) even without a full pass;
    # weakens = attribution never recovers across the sweep (894's FAIL stands). ----
    attribution_recovers = bool(any_gate_half or attribution_dose_recovers)
    if outcome_pass:
        direction = "supports"
        best_sigma = best_arm["remap_sigma"] if best_arm else None
        label = f"mech074d_attribution_gate_recovered_at_sigma_{best_sigma}"
    elif attribution_recovers:
        direction = "mixed"
        label = "mech074d_attribution_recovers_with_sigma_no_full_pass"
    else:
        direction = "weakens"
        label = "mech074d_attribution_gate_vacuous_across_sigma_sweep"

    combination_rule = (
        "SWEEP over bla_remap_pe_sigma_threshold in "
        f"{REMAP_SIGMA_SWEEP}. For EACH ON sigma arm, criterion Ck is MET when it holds "
        "on >= seeds_needed of that arm's GREEN (gate-passing) seeds; the arm PASSES iff "
        "C1 AND C2 AND C3 AND C4 (a plain AND within an arm). RUN PASS iff ANY ON sigma "
        "arm passes (an OR over the sweep -- the recalibrated threshold recovers the "
        "mechanism). C1+C2 are the ATTRIBUTION-GATE half (Moita 2004 dissociation); "
        "C3+C4 the PARTIALITY half. Direction: supports = some arm passes; mixed = the "
        "attribution gate RECOVERS with sparsification (C1 AND C2 met at some sigma, or "
        "mean selectivity rises monotonically across the sweep with the top sigma "
        "clearing C1 on >= seeds_needed green seeds) without a full pass; weakens = "
        "attribution never recovers (894's FAIL stands as a real weakening, not a "
        "calibration artifact)."
    )

    return {
        "outcome": "PASS" if outcome_pass else "FAIL",
        "evidence_direction": direction,
        "evidence_direction_per_claim": {"MECH-074d": direction},
        "interpretation_label": label,
        "n_seeds": n_seeds_run,
        "seeds_needed": int(seeds_needed),
        # any-arm criterion booleans (OR over the sweep) for the criteria[] list
        "c1_attribution_selectivity_met": c1_met_any,
        "c2_context_differentiated_addressing_met": c2_met_any,
        "c3_partial_not_wholesale_met": c3_met_any,
        "c4_pe_spike_sparsity_met": c4_met_any,
        "outcome_pass": outcome_pass,
        "passing_arm_ids": [a["arm"] for a in passing_arms],
        "best_arm_id": best_arm["arm"] if best_arm else None,
        "best_arm_sigma": best_arm["remap_sigma"] if best_arm else None,
        "attribution_recovers": attribution_recovers,
        "attribution_dose_recovers": attribution_dose_recovers,
        "any_attribution_gate_half_met": any_gate_half,
        "any_partiality_half_met": any_partial_half,
        "dose_response": {
            "sigmas": dr_sigmas,
            "mean_attr_mass_excess": dr_mass,
            "mean_fire_fraction": dr_fire,
            "spearman_mass_excess_vs_sigma": spearman_mass_vs_sigma,
            "spearman_fire_fraction_vs_sigma": spearman_fire_vs_sigma,
        },
        "per_arm": per_arm,
        "combination_rule": combination_rule,
        "per_seed": per_arm_per_seed,
    }


def main() -> Dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    t0 = time.perf_counter()

    seeds = SEEDS[:1] if args.dry_run else SEEDS

    # Design-time refusal: prove no arm faces a precondition it cannot satisfy in a way
    # that would silently vacate the run (the V3-EXQ-785 check, BEFORE compute).
    assert_no_structurally_unsatisfiable_gate(
        PRECONDITION_SPECS, [_arm_ctx(a) for a in ARMS]
    )

    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        for arm in ARMS:
            rows.append(_run_cell(seed, arm, dry=args.dry_run))

    ev = _evaluate(rows)

    # ---- Engagement assertions. Under --dry-run these are the cheap proof that the
    # decisive readout is non-trivially exercised BEFORE the full seed x arm grid is
    # committed. TWO tiers, because this is a sweep and the smoke's tiny episodes cannot
    # exercise the WHOLE sweep's range (a 15-step episode barely warms the PE EMA, so the
    # higher-sigma arms may still fire at smoke scale even though they sparsify at full
    # scale -- see the ROOT CAUSE docstring). Tier-1 (ENFORCED on --dry-run) is scale-
    # invariant. Tier-2 (REPORTED always, enforced only on a full run) is the sweep's own
    # dose-response, which needs full-scale episodes to show.
    on_rows = [r for r in rows if r["remap_live"]]
    off_rows = [r for r in rows if not r["remap_live"]]
    lowest_sigma = min(REMAP_SIGMA_SWEEP)
    low_arm_rows = [
        r for r in on_rows if abs(float(r["remap_sigma"]) - lowest_sigma) < 1e-9]
    diff_end_by_arm = {
        r["arm"]: float(r["slot_diff_std_episode_end_mean"]) for r in rows}
    fire_fraction_by_arm = {r["arm"]: float(r["fire_fraction"]) for r in rows}
    on_fire_fracs = [float(r["fire_fraction"]) for r in on_rows]
    engagement = {
        # -- Tier 1: scale-invariant, ENFORCED on --dry-run --
        "remap_fires_in_lowest_sigma_arm": all(
            int(r["n_remap_events"]) >= 1 for r in low_arm_rows) if low_arm_rows
            else False,
        "remap_silent_in_control_arm": all(
            int(r["n_remap_events"]) == 0 for r in off_rows) if off_rows else False,
        "store_differentiated_at_window_start": all(
            float(r["slot_diff_std_baseline"]) > SLOT_DIFF_STD_FLOOR
            for r in rows),
        # The per-episode restore must actually put every episode back at the baseline;
        # if it silently failed, C1/C2 would drift back to being measured on a store the
        # remap had already flattened -- the exact defect the restore exists to remove.
        "episode_restore_effective": all(
            abs(float(r["slot_diff_std_episode_start_mean"])
                - float(r["slot_diff_std_baseline"])) < 1e-6
            for r in rows),
        "pe_varies": all(float(r["pe_magnitude_std"]) > PE_SPREAD_FLOOR for r in rows),
        "bla_ticked": all(int(r["n_bla_ticks"]) > 0 for r in rows),
        "dv_varies_across_arms": (
            len({round(v, 9) for v in diff_end_by_arm.values()}) >= 2),
        # -- Tier 2: the sweep's dose-response, REPORTED always, enforced on full run --
        # The manipulation (sigma) MUST move fire fraction across the ON arms, else the
        # sweep is a saturation fingerprint (skill rule: a DV bit-identical across swept
        # values is a clamp/floor absorbing the manipulation, not a null). Rounded to 6dp
        # so pure float noise does not read as variation.
        "fire_fraction_varies_across_sweep": (
            len({round(f, 6) for f in on_fire_fracs}) >= 2),
        "slot_diff_std_end_by_arm": diff_end_by_arm,
        "fire_fraction_by_arm": fire_fraction_by_arm,
        "attr_mass_excess_by_arm": {
            r["arm"]: float(r["attr_mass_excess_mean"]) for r in rows},
    }
    print("[smoke] engagement: " + json.dumps(engagement), flush=True)
    if args.dry_run:
        # Tier-1 only on the smoke -- the tiny-episode dry-run cannot reach the higher
        # sigmas' sparsification regime, so fire_fraction_varies_across_sweep is checked
        # (and enforced) on the full run, not here.
        tier1 = {
            "remap_fires_in_lowest_sigma_arm",
            "remap_silent_in_control_arm",
            "store_differentiated_at_window_start",
            "episode_restore_effective",
            "pe_varies",
            "bla_ticked",
            "dv_varies_across_arms",
        }
        failed = [k for k in tier1
                  if isinstance(engagement.get(k), bool) and not engagement[k]]
        if failed:
            raise AssertionError(
                "dry-run engagement check failed: " + ", ".join(failed)
                + " -- the decisive readout is not properly exercised; fix the driver "
                  "before queuing the full grid"
            )
    elif not engagement["fire_fraction_varies_across_sweep"]:
        # Full run whose sweep did not move fire fraction: the manipulation was absorbed
        # (a clamp / env constraint), so the sweep is not evidence of anything. Fail loud
        # rather than record a flat sweep as a MECH-074d verdict.
        raise AssertionError(
            "full-run sweep engagement failed: fire_fraction_varies_across_sweep is "
            "False -- fire fraction is bit-identical across the swept sigmas, so the "
            "PE-sigma manipulation was absorbed (clamp / env constraint) rather than "
            "sparsifying the gate; this is a saturation fingerprint, not a null result"
        )

    aggregate = aggregate_arm_gates([r["arm_gate"] for r in rows])
    criteria_by_arm = {
        **{arm_id: [
            "C1_attribution_selectivity",
            "C2_context_differentiated_addressing",
            "C4_pe_spike_sparsity",
        ] for arm_id in ON_ARM_IDS},
        ARM_OFF: ["C3_partial_not_wholesale"],
    }
    non_degenerate_by_criterion = arm_criteria_non_degenerate(criteria_by_arm, aggregate)

    # Self-route: if NO ON sigma arm has any green cell the mechanism was not exercised
    # well enough at any threshold to judge the claim -- a REQUEUE, never a substrate
    # verdict.
    label = ev["interpretation_label"]
    direction = ev["evidence_direction"]
    per_claim = dict(ev["evidence_direction_per_claim"])
    on_gate_green = any(
        bool(r["arm_gate"].get("gate_green")) for r in rows if r["remap_live"])
    if not on_gate_green:
        label = "substrate_not_ready_requeue"
        direction = "unknown"
        per_claim = {"MECH-074d": "unknown"}

    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"

    full_config: Dict[str, Any] = {
        "threat_env_kwargs": THREAT_ENV_KWARGS,
        "neutral_env_kwargs": NEUTRAL_ENV_KWARGS,
        "arms": ARMS,
        "p0_episodes": P0_EPISODES if not args.dry_run else 3,
        "measure_episodes": (
            MEASURE_EPISODES if not args.dry_run else 2 * MIN_WINDOWS_PER_CONTEXT),
        "steps_per_episode": STEPS_PER_EPISODE if not args.dry_run else 15,
        "episodes_per_run": (
            TOTAL_EPISODES if not args.dry_run else 3 + 2 * MIN_WINDOWS_PER_CONTEXT),
        "measurement_baseline_restore": (
            "ContextMemory restored to the end-of-P0 snapshot at the start of EVERY "
            "measurement episode, identically in ALL arms"
        ),
        "remap_sigma_sweep": REMAP_SIGMA_SWEEP,
        "supersedes": SUPERSEDES,
        "world_dim": WORLD_DIM,
        "self_dim": SELF_DIM,
        "harm_dim": HARM_DIM,
        "harm_a_dim": HARM_A_DIM,
        "use_e2_harm_a": True,
        "bla_arousal_threshold_on": BLA_AROUSAL_THRESHOLD_ON,
        "bla_arousal_peak": BLA_AROUSAL_PEAK,
        "bla_window_steps": BLA_WINDOW_STEPS,
        "bla_remap_code_fraction": BLA_REMAP_CODE_FRACTION,
        "bla_context_remap_blend": BLA_CONTEXT_REMAP_BLEND,
        "bla_remap_requires_attribution": True,
        "remap_sigma_p0_all_arms": P0_REMAP_SIGMA,
        "thresholds": {
            "MIN_REMAP_EVENTS": MIN_REMAP_EVENTS,
            "SLOT_DIFF_STD_FLOOR": SLOT_DIFF_STD_FLOOR,
            "PE_SPREAD_FLOOR": PE_SPREAD_FLOOR,
            "MIN_WINDOWS_PER_CONTEXT": MIN_WINDOWS_PER_CONTEXT,
            "ATTR_MASS_EXCESS_MARGIN": ATTR_MASS_EXCESS_MARGIN,
            "CONTEXT_JACCARD_GAP_MARGIN": CONTEXT_JACCARD_GAP_MARGIN,
            "SLOT_DIFF_RATIO_FLOOR": SLOT_DIFF_RATIO_FLOOR,
            "FIRE_FRAC_CEIL": FIRE_FRAC_CEIL,
            "SEEDS_PASS_MIN": SEEDS_PASS_MIN,
        },
        "dry_run": bool(args.dry_run),
    }

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "claim_ids": CLAIM_IDS,
        "claim_ids_tested": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": ts,
        "outcome": ev["outcome"],
        "evidence_class": "exp:simulation",
        "evidence_direction": direction,
        "evidence_direction_per_claim": per_claim,
        "supersedes": SUPERSEDES,
        "dispatch_mode": "targeted_probe",
        "seed_policy": "distinct_seeds",
        "experiment_purpose_note": (
            "Recalibrated-threshold RETEST of MECH-074d (supersedes V3-EXQ-894, "
            "FAIL/weakens). Sweeps bla_remap_pe_sigma_threshold over "
            f"{REMAP_SIGMA_SWEEP} to test the 894-autopsy over-firing/dilution "
            "hypothesis: whether sparsifying an over-permissive gate recovers the "
            "attribution selectivity that read vacuous at the SD-035 default sigma=1.0. "
            "Wall-independent: every readout is internal to the ContextMemory store the "
            "remap writes to and the BLA gate itself; no behavioural outcome is read, so "
            "no downstream wall can gate the result."
        ),
        "acceptance": ev,
        "arm_results": rows,
        "per_seed_results": ev["per_seed"],
        "per_arm_results": ev["per_arm"],
        "dose_response": ev["dose_response"],
        "thresholds": full_config["thresholds"],
        "interpretation": {
            "label": label,
            "preconditions": aggregate["adjudication_preconditions"],
            "criteria_non_degenerate": non_degenerate_by_criterion,
            "combination_rule": ev["combination_rule"],
        },
        "criteria": [
            {"name": "C1_attribution_selectivity", "load_bearing": True,
             "passed": bool(ev["c1_attribution_selectivity_met"]),
             "claim": "MECH-074d"},
            {"name": "C2_context_differentiated_addressing", "load_bearing": True,
             "passed": bool(ev["c2_context_differentiated_addressing_met"]),
             "claim": "MECH-074d"},
            {"name": "C3_partial_not_wholesale", "load_bearing": True,
             "passed": bool(ev["c3_partial_not_wholesale_met"]),
             "claim": "MECH-074d"},
            {"name": "C4_pe_spike_sparsity", "load_bearing": False,
             "passed": bool(ev["c4_pe_spike_sparsity_met"]),
             "claim": "MECH-074d"},
        ],
        "per_arm_gate": aggregate["per_arm_gate"],
        "engagement_checks": engagement,
        "non_degenerate": bool(aggregate["non_degenerate"] and on_gate_green),
        "degeneracy_reason": (
            aggregate["degeneracy_reason"] if aggregate["degeneracy_reason"]
            else ("" if on_gate_green else
                  "no ON sigma arm had a green cell: the remap mechanism was not "
                  "exercised well enough at any threshold to judge MECH-074d; "
                  "self-routed substrate_not_ready_requeue")
        ),
        "summary": (
            f"{QUEUE_ID} MECH-074d BLA remap attribution sigma-sweep {REMAP_SIGMA_SWEEP}: "
            f"outcome={ev['outcome']} direction={direction} "
            f"passing_arms={ev['passing_arm_ids']} best_arm={ev['best_arm_id']} "
            f"attribution_recovers={ev['attribution_recovers']} "
            f"(any-arm C1={ev['c1_attribution_selectivity_met']} "
            f"C2={ev['c2_context_differentiated_addressing_met']} "
            f"C3={ev['c3_partial_not_wholesale_met']} "
            f"C4={ev['c4_pe_spike_sparsity_met']}) "
            f"over {ev['n_seeds']} seed(s); label={label}."
        ),
    }

    out_path = write_flat_manifest(
        manifest,
        dry_run=args.dry_run,
        config=full_config,
        seeds=seeds,
        script_path=Path(__file__),
        started_at=t0,
        z_goal_stream_stats=_ZG.stats(),
    )

    print(json.dumps({
        "run_id": run_id,
        "outcome": manifest["outcome"],
        "evidence_direction": manifest["evidence_direction"],
        "label": label,
        "manifest": str(out_path),
    }, indent=2))
    print(f"Result written to: {out_path}", flush=True)

    _outcome_raw = str(manifest["outcome"]).upper()
    return {
        "outcome": _outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        "manifest_path": out_path,
        "dry_run": bool(args.dry_run),
    }


if __name__ == "__main__":
    result = main()
    emit_outcome(
        outcome=result["outcome"],
        manifest_path=result["manifest_path"],
        dry_run=result["dry_run"],
    )
    raise SystemExit(0)
