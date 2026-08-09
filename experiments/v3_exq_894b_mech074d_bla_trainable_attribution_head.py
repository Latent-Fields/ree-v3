#!/opt/local/bin/python3
"""
V3-EXQ-894b -- MECH-074d BLA remap attribution: TRAINABLE ATTRIBUTION HEAD vs the fixed
               non-trainable rule (WALL-INDEPENDENT, representation-level DV).

POST-SUBSTRATE RETEST. Routed from the confirmed autopsy
``failure_autopsy_V3-EXQ-894a_2026-08-08`` (/failure-autopsy -> /implement-substrate ->
here). This driver is 894a with ONE design change: the swept dimension is retired and
the ATTRIBUTION IMPLEMENTATION becomes the arm axis. Everything else -- the DV, the
matched-baseline restore, the OFF drift control, the readiness gate, the DV-symmetry
scoping, the C1-C4 thresholds -- is preserved unchanged so all three runs are directly
comparable.

DOES NOT SUPERSEDE 894 OR 894a. Both are valid evidence about the non-trainable rule and
are precisely what motivated the substrate change under test here; nothing about them is
invalidated and nothing is de-weighted. ``SUPERSEDES`` is None on purpose.

WHY THIS RUN, AND WHAT IS ALREADY CLOSED.
Two independent experiments have now failed on the same criteria:

  V3-EXQ-894   FAIL/weakens. C1 (attribution_selectivity) and C2
               (context_differentiated_addressing) -- the Moita 2004 dissociation --
               fail on 2/3 seeds. Flagged a testable confound: fire-fraction and
               attribution-selectivity are INVERSELY correlated across seeds, i.e. an
               over-permissive gate might be diluting a real signal.
  V3-EXQ-894a  FAIL/weakens. Swept ``bla_remap_pe_sigma_threshold`` over
               [1.0, 1.5, 2.0, 2.5] to test exactly that confound, and REFUTED it:

                   sigma  C1 ok/3  C2 ok/3  mean mass_excess  mean fire_frac
                   1.0    1        1        0.0847            0.637
                   1.5    1        0        0.0773            0.402
                   2.0    1        0        0.0573            0.243
                   (2.5)  0        0        0.0074            0.086   [red/unscored]

               Decisive: spearman(mass_excess, sigma) = -1.0 AND
               spearman(fire_fraction, sigma) = -1.0. Dilution predicts selectivity
               should RISE as fire-fraction falls; instead both fall together. The
               per-seed ranking (45 selective, 43 context-blind-DETERMINISTIC with an
               exactly-zero jaccard gap at every threshold, 42 null) is stable at all
               four thresholds -- structural, not sampling noise.

So the PE-threshold dimension is CLOSED and is deliberately not re-swept here. It is
held at the SD-035 default 1.0 in every ON arm.

THE HYPOTHESIS UNDER TEST.
The confirmed autopsy's diagnosis is ``competence_implementation_gap``, and unusually it
is one the claim made about ITSELF in advance. MECH-074d's 2026-04-21 registration text
reads: "For the initial non-trainable pass this is approximated by selecting codes whose
contribution to the harm-PE exceeds a threshold; a learnable attribution head is
deferred." Moita 2004's dissociation DEVELOPS OVER TRAINING -- real BLA attribution is a
learned cue-outcome association -- so a fixed, context-agnostic threshold rule has no
mechanism by which it could become more selective with experience. That deferred second
pass landed 2026-08-09 as ``ree_core/amygdala/attribution_head.BLAAttributionHead``.

H1: with a LEARNED attribution feeding the same gate, C1 and C2 reach >= 2/3 seeds at
    some operating point -- the thresholds the fixed rule never reached at any sigma.
H0 (the outcome that would falsify the autopsy's own diagnosis): the learned head scores
    no better than the fixed rule, i.e. trainability was not the missing ingredient and
    the defect is deeper than implementation completeness.

ARMS (4 arms x 3 seeds; sigma FIXED at 1.0 in every ON arm).

  ARM_REMAP_OFF        sigma 1e9, gate can never open. The drift control and the C3
                       denominator. Unchanged from 894a.
  ARM_HEAD_FIXED       legacy non-trainable proxy. A WITHIN-RUN REPLICATION of 894a's
                       ARM_SIGMA_10, and the matched control that makes the trained
                       arms' numbers interpretable on THIS env/seed/substrate rather
                       than against a figure from a different run. If this arm PASSES
                       the attribution half, 894/894a did not replicate and the whole
                       comparison is void -- reported explicitly as
                       ``fixed_unexpectedly_passed``, never silently absorbed.
  ARM_HEAD_TRAINED     BLAAttributionHead, trained during P0 only (~1200 ticks).
  ARM_HEAD_TRAINED_LONG  same head, plus P1_HEAD_EPISODES extra gate-shut training
                       episodes (~2400 ticks total). This arm exists because the
                       substrate build measured the head to be UNDER-TRAINED rather
                       than converged at ~300 ticks (normalised attention entropy still
                       0.99-1.00). Two budgets turn "the head needs experience" from an
                       excuse available after a null into a claim the run TESTS: if
                       SHORT and LONG score the same, budget is not the limiting factor
                       and that is itself the finding.

WHAT THE SUBSTRATE PRE-CHECK ALREADY SHOWED, AND WHY IT IS NOT EVIDENCE.
During the 2026-08-09 build, a 3-seed 300-tick harness with the same matched-baseline
restore measured (legacy -> trainable): seed 42 mass_excess 0.034 -> 0.115; seed 43
jaccard_gap 0.010 -> 0.059; seed 45 jaccard_gap 0.005 -> 0.227. C2 moved 0/3 -> 2/3
seeds over the 0.05 margin, C1 0/3 -> 1/3. That is a smoke test at ~1/20 of this run's
training budget, with no precondition gate, no drift control and no pre-registration --
it is why this experiment is worth running, and it is NOT a result. Recorded here so a
reader of the manifest knows the prior, and so a null here is legible as a real
disagreement with the pre-check rather than a surprise.

MEASUREMENT (unchanged from 894a -- see that driver for the full derivation).
Alternating threat/neutral episodes. P0 warms the encoder with the gate SHUT in every
arm; the end-of-P0 ContextMemory store is snapshotted and RESTORED at the start of every
measurement episode, identically in all arms, so each episode is an independent
replicate from the same differentiated baseline. The measurement window runs with the
encoder frozen AND the attribution head frozen (P2 must read the weights P0/P1
produced). Attribution vectors are captured by wrapping
``REEAgent._get_context_memory_code_contributions``, which is the single entry point
BOTH implementations dispatch through -- so the recorded vector is the one the gate
actually consumed, in either arm, computed before the remap mutated the store.

  C1 attribution_selectivity  [LOAD-BEARING] mass on the k selected codes minus chance
                              k/n; margin 0.05.
  C2 context_differentiated_addressing  [LOAD-BEARING] within-context minus
                              cross-context Jaccard of the target set; margin 0.05. A
                              constant target set scores exactly 0.0 by construction.
  C3 partial_not_wholesale    [LOAD-BEARING] ON-arm end-of-episode slot differentiation
                              as a fraction of the matched OFF control's; floor 0.5.
  C4 pe_spike_sparsity        [not load-bearing] fire fraction <= 0.25.

RUN PASS iff a TRAINABLE arm meets C1 AND C2 AND C3 AND C4 on >= 2/3 of its green seeds.
ARM_HEAD_FIXED passing is NOT a pass for this question.

A PRECONDITION THIS DESIGN INHERITS, AND WHY THE RESTORE IS LOAD-BEARING HERE TOO.
Measured during the substrate build: under the legacy ContextMemory write path the slot
bank homogenises to off-diagonal cosine 1.0000 (slot norm 5.64 = ||0.5*ones(128)||, the
documented V3-EXQ-436c sigmoid-midpoint payload collapse) within ~24 episodes. Sixteen
identical slots cannot be differentiated by ANY attribution rule, trainable or not, so
without the per-episode restore this run would measure the write path rather than the
head. ``slot_differentiation_at_window_start`` is the precondition that enforces it and
``episode_restore_effective`` the engagement check that proves it fired.

ENGAGEMENT CHECKS THAT MATTER MOST HERE (beyond 894a's).
  trainable_heads_trained_past_warmup -- a head still inside its 200-step warmup falls
      back to the LEGACY proxy, which would silently make the "trained" arms duplicates
      of ARM_HEAD_FIXED. This is the single most likely way this design fails quietly,
      so a full run raises rather than reporting a vacuous null.
  baseline_store_matched_across_arms -- the head's optimiser must not perturb the
      encoder. Its inputs are detached and it owns its own parameters, so the post-P0
      store should be identical across arms at each seed. Asserted, not assumed.
  attr_mass_excess_varies_across_head_arms -- the anti-saturation rule 894a applied to
      its sweep: if the DV is bit-identical across arms the manipulation was absorbed
      and the contrast is an instrument fingerprint, not a null.

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

EXPERIMENT_TYPE = "v3_exq_894b_mech074d_bla_trainable_attribution_head"
QUEUE_ID = "V3-EXQ-894b"
# NOT a supersession. 894/894a are VALID evidence about the non-trainable rule -- they
# are what motivated the substrate change this run tests. Nothing about them is
# invalidated, so nothing is de-weighted.
SUPERSEDES = None
PRIOR_RUNS = [
    "v3_exq_894_mech074d_bla_remap_attribution_selectivity_20260808T005219Z_v3",
    "v3_exq_894a_mech074d_bla_remap_attribution_selectivity_20260808T101157Z_v3",
]
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
# The LONGEST arm (ARM_HEAD_TRAINED_LONG) is what the queue must be sized on -- the
# other arms are shorter by P1_HEAD_EPISODES. Defined after P1_HEAD_EPISODES below.

# BLA calibration -- identical across arms; ONLY the attribution implementation
# differs, and (in ARM_REMAP_OFF) the PE sigma threshold.
BLA_AROUSAL_THRESHOLD_ON = 0.4
BLA_AROUSAL_PEAK = 0.7
BLA_WINDOW_STEPS = 5
BLA_REMAP_CODE_FRACTION = 0.33     # SD-035 default (Moita 2004 ~30-35% overwrite)
BLA_CONTEXT_REMAP_BLEND = 0.5      # SD-035 default
# The PE sigma threshold is now HELD FIXED at the SD-035 default across every ON arm.
# 894a already swept it over [1.0, 1.5, 2.0, 2.5] and established that C1/C2 do not
# respond (Spearman -1.0, mass-excess falling WITH fire-fraction rather than diverging).
# Re-sweeping it here would re-spend compute on a closed question and confound the one
# dimension this run does change.
REMAP_SIGMA_ON = 1.0
REMAP_SIGMA_OFF = 1.0e9            # gate can never open (drift control)
P0_REMAP_SIGMA = 1.0e9             # all arms, during P0 only

# ---- The ONE dimension 894b changes: which attribution implementation feeds the gate.
HEAD_FIXED = "contribution_threshold"   # legacy non-trainable proxy (894/894a substrate)
HEAD_TRAINABLE = "trainable"            # BLAAttributionHead (landed 2026-08-09)

# Extra head-training episodes for the long arm, run AFTER P0 with the gate still shut
# and the encoder still training. Rationale, measured 2026-08-09 during the substrate
# build: at ~300 training ticks the head lifts C1 on one seed and C2 on two, but its
# attention is still near-uniform (normalised entropy 0.99-1.00) -- it is
# under-trained, not converged. P0 alone gives P0_EPISODES*STEPS_PER_EPISODE = 1200
# ticks; this arm doubles that so "the head needs experience" is TESTED rather than
# assumed. If the SHORT and LONG arms score the same, training budget is not the
# limiting factor and that is itself the finding.
P1_HEAD_EPISODES = 20

# Head hyperparameters -- identical in both trainable arms, so the SHORT/LONG contrast
# is purely training budget. warmup_steps must be comfortably below the smallest arm's
# training-tick count or the head would still be falling back to the legacy proxy at
# measurement time (which would silently turn the trained arms into duplicates of
# ARM_HEAD_FIXED -- checked explicitly in the engagement block).
ATTR_HEAD_WARMUP_STEPS = 200
ATTR_HEAD_LR = 1e-3
ATTR_HEAD_KEY_DIM = 16
ATTR_HEAD_ENTROPY_WEIGHT = 0.02
ATTR_HEAD_TEMPERATURE_INIT = 0.5

# Queue sizing = the longest arm.
TOTAL_EPISODES = P0_EPISODES + P1_HEAD_EPISODES + MEASURE_EPISODES

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
ARM_FIXED = "ARM_HEAD_FIXED"
ARM_TRAINED = "ARM_HEAD_TRAINED"
ARM_TRAINED_LONG = "ARM_HEAD_TRAINED_LONG"

# One OFF drift control + three ON arms. OFF FIRST so its per-seed differentiation
# baseline (the C3 denominator) is available when the ON arms evaluate.
#
# ARM_HEAD_FIXED is a WITHIN-RUN REPLICATION of 894a's ARM_SIGMA_10, not decoration: it
# is the matched control that makes the trained arms' numbers interpretable on THIS
# env/seed/substrate rather than against a figure from a different run. If it does not
# reproduce 894a's failure, the comparison is void and the engagement block says so.
ARMS: List[Dict[str, Any]] = [
    {"arm_id": ARM_OFF, "remap_sigma": REMAP_SIGMA_OFF, "remap_live": False,
     "head": HEAD_FIXED, "p1_head_episodes": 0},
    {"arm_id": ARM_FIXED, "remap_sigma": REMAP_SIGMA_ON, "remap_live": True,
     "head": HEAD_FIXED, "p1_head_episodes": 0},
    {"arm_id": ARM_TRAINED, "remap_sigma": REMAP_SIGMA_ON, "remap_live": True,
     "head": HEAD_TRAINABLE, "p1_head_episodes": 0},
    {"arm_id": ARM_TRAINED_LONG, "remap_sigma": REMAP_SIGMA_ON, "remap_live": True,
     "head": HEAD_TRAINABLE, "p1_head_episodes": P1_HEAD_EPISODES},
]
ON_ARM_IDS: List[str] = [ARM_FIXED, ARM_TRAINED, ARM_TRAINED_LONG]
# The arms whose attribution is LEARNED -- the ones the claim's second pass is about.
TRAINED_ARM_IDS: List[str] = [ARM_TRAINED, ARM_TRAINED_LONG]

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
        "head": str(arm["head"]),
        "p1_head_episodes": int(arm["p1_head_episodes"]),
    }


def _make_env(seed: int, kwargs: Dict[str, Any]) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **kwargs)


def _build_config(env: CausalGridWorldV2, arm: Dict[str, Any]) -> REEConfig:
    """Config is identical across arms EXCEPT bla_attribution_head, which IS the
    manipulation and must be set at construction time (the head is instantiated in
    REEAgent.__init__ and cannot be swapped in afterwards).

    The PE sigma threshold is NOT set here -- it is installed on agent.bla.config at
    the P0/measurement boundary exactly as in 894a, so P0 is bit-comparable across
    arms.

    Note the asymmetry this creates, and why it is acceptable: the trainable arms
    additionally run their head's optimiser during P0. That optimiser touches nothing
    the rest of the agent reads -- every input is detached and the head owns its own
    parameters -- so the encoder trajectory should be unaffected. That is ASSERTED,
    not assumed: the engagement block checks the post-P0 baseline store is identical
    across arms at each seed."""
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
        # ---- THE MANIPULATION ----
        bla_attribution_head=str(arm["head"]),
        bla_attr_head_warmup_steps=ATTR_HEAD_WARMUP_STEPS,
        bla_attr_head_lr=ATTR_HEAD_LR,
        bla_attr_head_key_dim=ATTR_HEAD_KEY_DIM,
        bla_attr_head_entropy_weight=ATTR_HEAD_ENTROPY_WEIGHT,
        bla_attr_head_temperature_init=ATTR_HEAD_TEMPERATURE_INIT,
        bla_attr_head_train=True,
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

    p1_head_eps = (2 if (dry and int(arm["p1_head_episodes"]) > 0)
                   else int(arm["p1_head_episodes"]))
    total_eps = p0_eps + p1_head_eps + meas_eps

    config_slice = {
        "arm_id": arm_id,
        "attribution_head": str(arm["head"]),
        "p1_head_episodes": p1_head_eps,
        "attr_head_warmup_steps": ATTR_HEAD_WARMUP_STEPS,
        "attr_head_lr": ATTR_HEAD_LR,
        "attr_head_key_dim": ATTR_HEAD_KEY_DIM,
        "attr_head_entropy_weight": ATTR_HEAD_ENTROPY_WEIGHT,
        "attr_head_temperature_init": ATTR_HEAD_TEMPERATURE_INIT,
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
        cfg = _build_config(threat_env, arm)
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

        # ---- Optional extra head-training phase: gate still SHUT, encoder still
        # training, so the ONLY thing it buys is more attribution-head experience. The
        # gate stays shut throughout, so no remap write can differ between the SHORT and
        # LONG arms before measurement -- both enter the measurement window with the
        # same KIND of store, differing only in how much the head has learned.
        if p1_head_eps > 0:
            _run_phase(
                agent, threat_env, neutral_env,
                num_episodes=p1_head_eps, steps_per_episode=steps, seed=seed,
                episode_offset=p0_eps, total_episodes=total_eps,
                train_mode=True, record=False, label=label + " [p1-head]",
            )

        # ---- Install the arm's gate, then measure with the encoder frozen. ----
        # The end-of-P0 store is the BASELINE every measurement episode is restored to,
        # in both arms identically (module docstring (b)).
        mem_baseline = cm.memory.data.detach().clone()
        slot_diff_baseline = _slot_diff_std(mem_baseline)
        if agent.bla is not None:
            agent.bla.config.remap_pe_sigma_threshold = float(arm["remap_sigma"])

        # FREEZE the head for the measurement window. _run_phase already passes
        # train_mode=False (so agent.training is False and the agent-side training hook
        # is skipped); this is the belt-and-braces second latch on the head's own
        # config. P2 must measure the weights P0/P1 produced, not weights still moving
        # while the DV is read.
        head_diag_pre_measure: Dict[str, Any] = {}
        if agent.bla_attribution_head is not None:
            head_diag_pre_measure = dict(agent.bla_attribution_head.diagnostics)
            agent.bla_attribution_head.config.train = False

        meas = _run_phase(
            agent, threat_env, neutral_env,
            num_episodes=meas_eps, steps_per_episode=steps, seed=seed,
            episode_offset=p0_eps + p1_head_eps, total_episodes=total_eps,
            train_mode=False, record=True, label=label,
            restore_base=mem_baseline,
        )
        _ZG.observe(agent)   # AFTER stepping -- reads the counters at call time

        summ = _summarise(
            meas, n_slots=n_slots, slot_diff_baseline=slot_diff_baseline,
        )

        head_diag_post = (
            dict(agent.bla_attribution_head.diagnostics)
            if agent.bla_attribution_head is not None else {})

        row: Dict[str, Any] = {
            "seed": int(seed),
            "arm": arm_id,
            "remap_sigma": float(arm["remap_sigma"]),
            "remap_live": bool(arm["remap_live"]),
            "attribution_head": str(arm["head"]),
            "p1_head_episodes": int(p1_head_eps),
            "head_is_trainable": bool(agent.bla_attribution_head is not None),
            # The head's own telemetry, banked generously so a successor can ask whether
            # a null was under-training, saturation, or a genuinely flat attribution
            # WITHOUT re-running the grid.
            "head_diagnostics_pre_measure": head_diag_pre_measure,
            "head_diagnostics_post_measure": head_diag_post,
            "head_n_updates": int(head_diag_post.get("n_updates", 0)),
            "head_is_warm": bool(head_diag_post.get("is_warm", False)),
            "head_warmup_fallbacks": int(head_diag_post.get("n_warmup_fallbacks", 0)),
            # Sourced from the PRE-measure snapshot on purpose. last_norm_entropy /
            # last_pred_loss are latches written by train_step, and BLAAttributionHead
            # .reset() clears them -- which the driver triggers at the start of every
            # measurement episode via agent.reset(). Since training is frozen for the
            # measurement window, the post-measure copies are always 0.0 and would
            # silently read as "attention fully collapsed" rather than "not recorded".
            # last_max_weight is written by attribute(), not train_step, so the POST
            # copy of that one is the live measurement-window value.
            "head_norm_entropy": float(
                head_diag_pre_measure.get("last_norm_entropy", 0.0)),
            "head_pred_loss": float(head_diag_pre_measure.get("last_pred_loss", 0.0)),
            "head_max_weight_at_measure": float(
                head_diag_post.get("last_max_weight", 0.0)),
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


def _evaluate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Head-arm evaluation. For each ON arm, tally C1-C4 over that arm's GREEN
    (gate-passing) seeds against the matched OFF drift control at the same seed; an arm
    PASSES iff all four are met on >= seeds_needed green seeds.

    The RUN passes iff a TRAINABLE arm passes -- ARM_HEAD_FIXED passing would NOT be a
    pass for this run's question, it would mean the within-run replication of 894a's
    failure did not reproduce and the comparison is void (flagged separately). Direction
    keys on whether the LEARNED head recovers the attribution gate (C1+C2, the Moita
    2004 dissociation) that the fixed rule could not reach at any of 894a's four
    thresholds."""
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
        arm_spec = next(a for a in ARMS if a["arm_id"] == arm_id)
        sigma = float(arm_spec["remap_sigma"])
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
                "attribution_head": str(arm_spec["head"]),
                "p1_head_episodes": int(arm_spec["p1_head_episodes"]),
                "head_n_updates": int(on.get("head_n_updates", 0)),
                "head_is_warm": bool(on.get("head_is_warm", False)),
                "head_norm_entropy": float(on.get("head_norm_entropy", 0.0)),
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
            "attribution_head": str(arm_spec["head"]),
            "p1_head_episodes": int(arm_spec["p1_head_episodes"]),
            "is_trainable_arm": bool(arm_id in TRAINED_ARM_IDS),
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

    # ---- Fixed-vs-trained comparison. This replaces 894a's sigma dose-response: the
    # swept dimension there is held constant here, and the contrast that matters is
    # LEARNED vs FIXED attribution at the same threshold, same seeds, same env. ----
    by_arm = {a["arm"]: a for a in per_arm}
    fixed = by_arm.get(ARM_FIXED)

    def _delta(arm_id: str, key: str) -> Optional[float]:
        """Trained-arm mean minus the fixed-rule mean on the same statistic.

        None (not 0.0) when either side has no green seed -- an absent comparison must
        not be reported as "no difference"."""
        a = by_arm.get(arm_id)
        if a is None or fixed is None:
            return None
        av, fv = a.get(key), fixed.get(key)
        if av is None or fv is None:
            return None
        return float(av) - float(fv)

    head_contrast = {
        arm_id: {
            "mass_excess_delta_vs_fixed": _delta(arm_id, "mean_attr_mass_excess"),
            "jaccard_gap_delta_vs_fixed": _delta(arm_id, "mean_jaccard_gap"),
            "fire_fraction_delta_vs_fixed": _delta(arm_id, "mean_fire_fraction"),
            "slot_diff_ratio_delta_vs_fixed": _delta(arm_id, "mean_slot_diff_ratio"),
            "c1_seeds_ok": by_arm[arm_id]["c1_seeds_ok"] if arm_id in by_arm else 0,
            "c2_seeds_ok": by_arm[arm_id]["c2_seeds_ok"] if arm_id in by_arm else 0,
        }
        for arm_id in TRAINED_ARM_IDS if arm_id in by_arm
    }

    # Did the within-run replication of 894a actually reproduce? If ARM_HEAD_FIXED
    # PASSES the attribution half, then the fixed rule is selective in this run and
    # 894/894a did not replicate -- which makes any trained-vs-fixed delta
    # uninterpretable. Reported explicitly rather than silently folded into the verdict.
    fixed_replicates_894a = bool(
        fixed is not None and fixed["n_green_seeds"] >= seeds_needed
        and not fixed["attribution_gate_half_met"])
    fixed_unexpectedly_passed = bool(
        fixed is not None and fixed["attribution_gate_half_met"])

    # Does MORE head training help? Compares the LONG arm against the SHORT one on the
    # attribution half. A null here with both trained arms flat is a different finding
    # from a null with the long arm clearly better -- the first says the head form is
    # wrong, the second says the training budget was.
    short_a, long_a = by_arm.get(ARM_TRAINED), by_arm.get(ARM_TRAINED_LONG)
    training_budget_helps = None
    if short_a is not None and long_a is not None:
        sm, lm = short_a["mean_attr_mass_excess"], long_a["mean_attr_mass_excess"]
        sj, lj = short_a["mean_jaccard_gap"], long_a["mean_jaccard_gap"]
        if None not in (sm, lm, sj, lj):
            training_budget_helps = bool(
                (float(lm) > float(sm)) and (float(lj) >= float(sj)))

    trained_arms = [a for a in per_arm if a["arm"] in TRAINED_ARM_IDS]
    any_trained_gate_half = any(a["attribution_gate_half_met"] for a in trained_arms)
    any_gate_half = any(a["attribution_gate_half_met"] for a in per_arm)
    any_partial_half = any(a["partiality_half_met"] for a in per_arm)

    # A run PASS requires a TRAINABLE arm to pass. ARM_HEAD_FIXED passing is not this
    # run's hypothesis (see fixed_unexpectedly_passed above).
    passing_arms = [a for a in per_arm
                    if a["arm_pass"] and a["arm"] in TRAINED_ARM_IDS]
    outcome_pass = bool(passing_arms)
    fixed_arm_passed = bool(fixed is not None and fixed["arm_pass"])

    scored = [a for a in per_arm if a["n_green_seeds"] >= 1]
    best_arm = (
        passing_arms[0] if passing_arms
        else (max(trained_arms, key=lambda a: (a["c1_seeds_ok"], a["c2_seeds_ok"]))
              if trained_arms else (scored[0] if scored else None)))

    # Any-arm criterion booleans (for the top-level criteria[] list + summary). An OR
    # over arms -- the per-arm booleans in per_arm[] carry the head-resolved detail.
    c1_met_any = any(a["c1_attribution_selectivity_met"] for a in per_arm)
    c2_met_any = any(a["c2_context_differentiated_addressing_met"] for a in per_arm)
    c3_met_any = any(a["c3_partial_not_wholesale_met"] for a in per_arm)
    c4_met_any = any(a["c4_pe_spike_sparsity_met"] for a in per_arm)

    # ---- Direction. supports = a trainable arm fully passes; mixed = the learned head
    # recovers the attribution HALF (C1 AND C2) without a full pass, or beats the fixed
    # rule on both attribution statistics; weakens = the learned head does no better
    # than the fixed rule, i.e. trainability was not the missing ingredient. ----
    beats_fixed_on_both = any(
        (hc["mass_excess_delta_vs_fixed"] or 0.0) > 0.0
        and (hc["jaccard_gap_delta_vs_fixed"] or 0.0) > 0.0
        for hc in head_contrast.values()
    )
    attribution_recovers = bool(any_trained_gate_half or beats_fixed_on_both)
    if outcome_pass:
        direction = "supports"
        label = f"mech074d_attribution_gate_recovered_by_trainable_head_{best_arm['arm'].lower()}"
    elif attribution_recovers:
        direction = "mixed"
        label = "mech074d_trainable_head_improves_attribution_no_full_pass"
    else:
        direction = "weakens"
        label = "mech074d_trainable_head_does_not_recover_attribution_gate"

    combination_rule = (
        "ARM AXIS = the attribution implementation feeding the MECH-074d remap gate, at "
        f"a FIXED bla_remap_pe_sigma_threshold of {REMAP_SIGMA_ON} (894a already swept "
        "that dimension and closed it, Spearman -1.0). Arms: ARM_HEAD_FIXED (legacy "
        "non-trainable proxy -- a WITHIN-RUN replication of 894a's ARM_SIGMA_10 and the "
        "matched control), ARM_HEAD_TRAINED (BLAAttributionHead, P0 training only), "
        "ARM_HEAD_TRAINED_LONG (same head, plus P1_HEAD_EPISODES extra gate-shut "
        "training episodes). For EACH ON arm, criterion Ck is MET when it holds on >= "
        "seeds_needed of that arm's GREEN (gate-passing) seeds; the arm PASSES iff C1 "
        "AND C2 AND C3 AND C4. RUN PASS iff a TRAINABLE arm passes -- ARM_HEAD_FIXED "
        "passing is NOT a pass for this question and instead voids the comparison "
        "(reported as fixed_unexpectedly_passed). C1+C2 are the ATTRIBUTION-GATE half "
        "(Moita 2004 dissociation); C3+C4 the PARTIALITY half. Direction: supports = a "
        "trainable arm passes; mixed = a trainable arm recovers C1 AND C2 without a "
        "full pass, or beats the fixed rule on BOTH attribution statistics; weakens = "
        "the learned head does no better than the fixed rule, i.e. trainability was not "
        "the missing ingredient and the diagnosis in "
        "failure_autopsy_V3-EXQ-894a_2026-08-08 is itself wrong."
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
        "best_arm_head": best_arm["attribution_head"] if best_arm else None,
        "attribution_recovers": attribution_recovers,
        "any_attribution_gate_half_met": any_gate_half,
        "any_trained_attribution_gate_half_met": any_trained_gate_half,
        "any_partiality_half_met": any_partial_half,
        "fixed_arm_passed": fixed_arm_passed,
        "fixed_replicates_894a": fixed_replicates_894a,
        "fixed_unexpectedly_passed": fixed_unexpectedly_passed,
        "training_budget_helps": training_budget_helps,
        "trained_beats_fixed_on_both_attribution_stats": beats_fixed_on_both,
        "head_contrast": head_contrast,
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
    fixed_rows = [r for r in on_rows if r["arm"] == ARM_FIXED]
    trained_rows = [r for r in on_rows if r["arm"] in TRAINED_ARM_IDS]
    diff_end_by_arm = {
        r["arm"]: float(r["slot_diff_std_episode_end_mean"]) for r in rows}
    fire_fraction_by_arm = {r["arm"]: float(r["fire_fraction"]) for r in rows}
    mass_excess_by_arm = {r["arm"]: float(r["attr_mass_excess_mean"]) for r in rows}
    on_mass_excess = [float(r["attr_mass_excess_mean"]) for r in on_rows]

    # Did the trainable arms' heads actually get USED at measurement time? A head still
    # inside warmup silently falls back to the legacy proxy, which would make the
    # "trained" arms duplicates of ARM_HEAD_FIXED and the whole contrast vacuous -- the
    # single most likely way this design fails quietly.
    heads_warm = all(bool(r.get("head_is_warm")) for r in trained_rows) if trained_rows \
        else False
    heads_trained_enough = all(
        int(r.get("head_n_updates", 0)) > ATTR_HEAD_WARMUP_STEPS for r in trained_rows
    ) if trained_rows else False
    # ... and the FIXED arm must have built no head at all.
    fixed_has_no_head = all(
        not bool(r.get("head_is_trainable")) for r in rows if r["arm"] not in
        TRAINED_ARM_IDS)

    # The head's optimiser must not have perturbed the encoder: the post-P0 baseline
    # store is the C1/C2 measurement substrate, and if it differs across arms at the
    # same seed then the arms are not measuring the same thing. Compared per seed.
    baseline_by_seed: Dict[int, List[float]] = {}
    for r in rows:
        baseline_by_seed.setdefault(int(r["seed"]), []).append(
            float(r["slot_diff_std_baseline"]))
    baseline_matched_across_arms = all(
        (max(v) - min(v)) < 1e-9 for v in baseline_by_seed.values())

    engagement = {
        # -- Tier 1: scale-invariant, ENFORCED on --dry-run --
        "remap_fires_in_fixed_arm": all(
            int(r["n_remap_events"]) >= 1 for r in fixed_rows) if fixed_rows else False,
        "remap_fires_in_trained_arms": all(
            int(r["n_remap_events"]) >= 1 for r in trained_rows) if trained_rows
            else False,
        "remap_silent_in_control_arm": all(
            int(r["n_remap_events"]) == 0 for r in off_rows) if off_rows else False,
        "store_differentiated_at_window_start": all(
            float(r["slot_diff_std_baseline"]) > SLOT_DIFF_STD_FLOOR for r in rows),
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
        "trainable_heads_are_warm": heads_warm,
        "fixed_arm_built_no_head": fixed_has_no_head,
        # -- Tier 2: REPORTED always, enforced only on a full run --
        # The manipulation (which attribution rule feeds the gate) MUST move the
        # attribution statistic across the ON arms, else the DV is saturated / clamped
        # and the contrast is a fingerprint of the instrument, not of the head. Same
        # anti-saturation rule 894a applied to its sigma sweep. Rounded to 6dp so pure
        # float noise does not read as variation.
        "attr_mass_excess_varies_across_head_arms": (
            len({round(f, 6) for f in on_mass_excess}) >= 2),
        "trainable_heads_trained_past_warmup": heads_trained_enough,
        "baseline_store_matched_across_arms": baseline_matched_across_arms,
        "slot_diff_std_end_by_arm": diff_end_by_arm,
        "fire_fraction_by_arm": fire_fraction_by_arm,
        "attr_mass_excess_by_arm": mass_excess_by_arm,
        "head_n_updates_by_arm": {
            r["arm"]: int(r.get("head_n_updates", 0)) for r in rows},
        "head_norm_entropy_by_arm": {
            r["arm"]: float(r.get("head_norm_entropy", 0.0)) for r in rows},
        "head_max_weight_at_measure_by_arm": {
            r["arm"]: float(r.get("head_max_weight_at_measure", 0.0)) for r in rows},
    }
    print("[smoke] engagement: " + json.dumps(engagement), flush=True)
    if args.dry_run:
        # Tier-1 only on the smoke. trainable_heads_are_warm is NOT in tier 1: the
        # dry-run runs 3 P0 episodes x 15 steps = 45 ticks against a 200-step warmup, so
        # the head CANNOT be warm at smoke scale and enforcing it here would fail every
        # smoke for a reason that is purely an artifact of the smoke's own episode
        # count. It is enforced on the full run below, where 1200+ P0 ticks make it a
        # real check.
        tier1 = {
            "remap_fires_in_fixed_arm",
            "remap_silent_in_control_arm",
            "store_differentiated_at_window_start",
            "episode_restore_effective",
            "pe_varies",
            "bla_ticked",
            "dv_varies_across_arms",
            "fixed_arm_built_no_head",
        }
        failed = [k for k in tier1
                  if isinstance(engagement.get(k), bool) and not engagement[k]]
        if failed:
            raise AssertionError(
                "dry-run engagement check failed: " + ", ".join(failed)
                + " -- the decisive readout is not properly exercised; fix the driver "
                  "before queuing the full grid"
            )
    else:
        # Full-run enforcement. Each of these, if False, makes the run's central contrast
        # uninterpretable rather than negative -- so fail loud instead of recording a
        # vacuous comparison as a MECH-074d verdict.
        full_failures = []
        if not engagement["trainable_heads_trained_past_warmup"]:
            full_failures.append(
                "trainable_heads_trained_past_warmup is False -- at least one trainable "
                f"arm finished with <= {ATTR_HEAD_WARMUP_STEPS} head updates, so its "
                "attribute() was still falling back to the LEGACY proxy and the arm is "
                "a duplicate of ARM_HEAD_FIXED, not a test of the trained head"
            )
        if not engagement["attr_mass_excess_varies_across_head_arms"]:
            full_failures.append(
                "attr_mass_excess_varies_across_head_arms is False -- the attribution "
                "statistic is bit-identical across the head arms, so the manipulation "
                "was absorbed rather than changing what the gate selects; a saturation "
                "fingerprint, not a null result"
            )
        if not engagement["baseline_store_matched_across_arms"]:
            full_failures.append(
                "baseline_store_matched_across_arms is False -- the post-P0 baseline "
                "store differs across arms at the same seed, so the head's optimiser "
                "perturbed the encoder trajectory and the arms are not measuring the "
                "same substrate (the detached-input guarantee is broken)"
            )
        if full_failures:
            raise AssertionError(
                "full-run engagement failed: " + " | ".join(full_failures))

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

    # Self-route: if NO ON arm has any green cell the mechanism was not exercised well
    # enough under any attribution implementation to judge the claim -- a REQUEUE, never
    # a substrate verdict.
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
        "p1_head_episodes_long_arm": P1_HEAD_EPISODES if not args.dry_run else 2,
        "measure_episodes": (
            MEASURE_EPISODES if not args.dry_run else 2 * MIN_WINDOWS_PER_CONTEXT),
        "steps_per_episode": STEPS_PER_EPISODE if not args.dry_run else 15,
        # Longest arm (ARM_HEAD_TRAINED_LONG) -- the runner sizes on the worst case.
        "episodes_per_run": (
            TOTAL_EPISODES if not args.dry_run
            else 3 + 2 + 2 * MIN_WINDOWS_PER_CONTEXT),
        "measurement_baseline_restore": (
            "ContextMemory restored to the end-of-P0 snapshot at the start of EVERY "
            "measurement episode, identically in ALL arms"
        ),
        "remap_sigma_on_all_on_arms": REMAP_SIGMA_ON,
        "attribution_head_arms": {a["arm_id"]: a["head"] for a in ARMS},
        "attr_head_hyperparameters": {
            "warmup_steps": ATTR_HEAD_WARMUP_STEPS,
            "lr": ATTR_HEAD_LR,
            "key_dim": ATTR_HEAD_KEY_DIM,
            "entropy_weight": ATTR_HEAD_ENTROPY_WEIGHT,
            "temperature_init": ATTR_HEAD_TEMPERATURE_INIT,
        },
        "supersedes": SUPERSEDES,
        "prior_runs_not_superseded": PRIOR_RUNS,
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
        "prior_runs_not_superseded": PRIOR_RUNS,
        "dispatch_mode": "targeted_probe",
        "seed_policy": "distinct_seeds",
        "experiment_purpose_note": (
            "POST-SUBSTRATE retest of MECH-074d. V3-EXQ-894 and V3-EXQ-894a both "
            "FAIL/weakens; 894a swept bla_remap_pe_sigma_threshold over "
            "[1.0,1.5,2.0,2.5] and REFUTED the over-firing/dilution alternative "
            "(Spearman -1.0, mass-excess falling WITH fire-fraction). The confirmed "
            "autopsy diagnosed competence_implementation_gap: the attribution "
            "computation was a fixed non-trainable rule, which MECH-074d's own "
            "2026-04-21 registration text had already flagged as a placeholder pending "
            "'a deliberate second pass'. That second pass landed 2026-08-09 "
            "(ree_core/amygdala/attribution_head.BLAAttributionHead). This run holds "
            f"sigma FIXED at {REMAP_SIGMA_ON} and makes the ATTRIBUTION IMPLEMENTATION "
            "the arm axis: legacy fixed rule (a within-run replication of 894a's "
            "ARM_SIGMA_10) vs the learned head at two training budgets. It does NOT "
            "supersede 894/894a -- those remain valid evidence about the "
            "non-trainable rule and are what motivated the substrate change. "
            "Wall-independent: every readout is internal to the ContextMemory store the "
            "remap writes to and the BLA gate itself; no behavioural outcome is read, so "
            "no downstream wall can gate the result."
        ),
        "acceptance": ev,
        "arm_results": rows,
        "per_seed_results": ev["per_seed"],
        "per_arm_results": ev["per_arm"],
        "head_contrast": ev["head_contrast"],
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
                  "no ON arm had a green cell: the remap mechanism was not exercised "
                  "well enough under any attribution implementation to judge "
                  "MECH-074d; self-routed substrate_not_ready_requeue")
        ),
        "summary": (
            f"{QUEUE_ID} MECH-074d trainable BLA attribution head vs fixed rule "
            f"(sigma held at {REMAP_SIGMA_ON}): "
            f"outcome={ev['outcome']} direction={direction} "
            f"passing_arms={ev['passing_arm_ids']} best_arm={ev['best_arm_id']} "
            f"attribution_recovers={ev['attribution_recovers']} "
            f"fixed_replicates_894a={ev['fixed_replicates_894a']} "
            f"training_budget_helps={ev['training_budget_helps']} "
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
