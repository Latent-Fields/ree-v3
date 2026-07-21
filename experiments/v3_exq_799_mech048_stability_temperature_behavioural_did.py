"""V3-EXQ-799 -- MECH-048 mu-overlay: does ENVIRONMENTALLY-DRIVEN stability change
BEHAVIOUR through the mode-prior softmax temperature, or only the entropy arithmetic?

SLEEP DRIVER: N/A (waking measurement phase; no sleep loop. The STABLE regime calls
agent.enter_offline_mode() as a REST event to reset PCC offline-recency -- that is the
MECH-092 quiescence integration point, not a SleepLoopManager cycle.)

PURPOSE. Validation spike for the SD-032d amendment landed 2026-07-21 (substrate_queue
row sd_032d_mu_kappa_mode_prior_overlays, status implemented_pending_validation).
experiment_purpose = "diagnostic": this yields ROUTING, not a verdict. See "WHY
DIAGNOSTIC" below -- tagging it evidence would put a structurally-vacuous first entry
against a claim currently at exp=0, which is exactly what must not happen.

WHAT LANDED, AND THE HOLE IT LEFT
---------------------------------
MECH-048 asserts opponent stability overlays modulating BOTH (a) mode-prior
sharpness/entropy AND (b) switching pressure. Before 2026-07-21 the mu-analogue
(pcc_stability, SD-032d) reached ONLY the switch-threshold multiplier --
affinity_weights["pcc_stability"] was deliberately empty -- so H(operating_mode) was
EXACTLY invariant under mu (0.167605 at pcc_stability 0.0, 1.0 and 3.0, identical to
6 dp, measured by live execution in session practical-volhard-2c1876).

The build couples mu into the mode-prior softmax temperature in the form given by the
claim's own source (docs/thoughts/2026-02-11_some_control_plane_maths_hypotheses.md:63):

    tau = softmax_temperature * exp(alpha_kappa*aic_salience - alpha_mu*pcc_stability)

Config: REEConfig.salience_use_stability_temperature (default False, bit-identical when
off), salience_temperature_mu_alpha (1.0), salience_temperature_kappa_alpha (0.0),
salience_temperature_exponent_clip (4.0). SalienceCoordinator.tick() now returns
effective_temperature and mode_entropy.

THE VACUITY CONSTRAINT THAT DEFINES THIS DESIGN (read before changing anything)
------------------------------------------------------------------------------
With mu injected DIRECTLY into the temperature, "entropy falls as mu rises" is an
ARITHMETIC IDENTITY of the coupling -- true on any substrate, whether or not MECH-048
is true. That is the DV-symmetry vacuity class of
failure_autopsy_V3-EXQ-604c_2026-07-20 section 3, and it is why the predecessor
V3-EXQ-683 was HELD at code review (its C1/C2 set pcc_stability values and read
H(softmax(scores/T)) -- delta fixed before the run).

So this experiment does two things differently, and BOTH are load-bearing:

  (1) mu is driven ONLY from UPSTREAM ENVIRONMENTAL STATE. The script never writes
      pcc_stability and never calls update_signal("pcc_stability", ...). It changes the
      WORLD (resource density, hazard density) and the agent's own rest schedule; the
      per-step task outcome fed to agent.note_task_outcome() is read from per-step
      DELTAS of the env's total_benefit / total_harm counters (benefit delta > 0 ->
      1.0, harm delta > HARM_EVENT_DELTA -> 0.0, else neutral 0.5 -- see
      HARM_EVENT_DELTA for why the raw harm_signal is the wrong source and actually
      INVERTS the manipulation). PCCAnalog then derives mu from success-EMA +
      drive_level fatigue +
      steps-since-offline. Whether the environment actually moves mu is NOT assumed --
      it is precondition P1, and no experiment has ever exercised use_pcc_analog
      (0 manifests mention it), so P1 is the real substrate-readiness question.

  (2) The load-bearing DV is BEHAVIOURAL and TRAJECTORY-LEVEL, and it is read as a
      DIFFERENCE-IN-DIFFERENCES against a coupling-OFF control.

WHY THE DiD IS THE ANTI-VACUITY DEVICE
--------------------------------------
mu moves with the environment in BOTH coupling arms -- it always did, via the
stability_scaling threshold multiplier and via drive_level's own affinity weight
(affinity_weights["drive_level"]["external_task"]). So a plain env -> behaviour main
effect exists with the coupling OFF and is NOT evidence for the amendment.

The coupling-OFF arm therefore carries every non-temperature env -> behaviour path, and
the interaction

    interaction = [breadth(VOLATILE,ON)  - breadth(STABLE,ON)]
                - [breadth(VOLATILE,OFF) - breadth(STABLE,OFF)]

isolates the behavioural contribution of the temperature channel alone. Nothing in the
arithmetic fixes its sign or magnitude: it is zero unless the entropy modulation
actually PROPAGATES into action. That propagation is a genuinely open substrate
question, and it could well be zero (see THE LIVE CHANNEL).

DV-SYMMETRY DECLARATION (mandatory per arm; skill Step 3 "DV-SYMMETRY INVARIANCE")
---------------------------------------------------------------------------------
The load-bearing DV is state-visitation entropy over the realized trajectory (C1) and
committed-action-class entropy over FRESH selections (C2). Symmetry group of both: they
are set-aggregates (permutation-invariant) over the realized state/action sequence, and
they are invariant under any relabelling of cells/action-classes.

The manipulation is NOT invariant under that group, for a reason that must be checked
rather than assumed, so it is precondition P3:

  * The temperature change alters operating_mode p, which enters behaviour through
    write_gate(target) = sum_m p_m * w[target][m] -- a SOFT read of p, so it does NOT
    require any mode switch or contested-mode occupancy (see THE LIVE CHANNEL).
  * write_gate("e3_policy") MULTIPLIES the per-candidate dACC score_bias VECTOR
    (agent.py:5790, gated by salience_apply_to_dacc_bias). Scaling ONE additive term
    among several re-weights it against the other score terms and CAN move the E3
    argmax. It is NOT a broadcast additive constant (which would cancel in argmax and
    softmax -- the 604c failure) and NOT a monotone rescaling of the total score.
  * That escape holds ONLY IF the dACC bias genuinely varies across candidates. A bias
    that is uniform across candidates IS a broadcast constant and WOULD be vacuous. So
    P3 asserts the CROSS-CANDIDATE RANGE of the dACC bias, which is the SAME statistic
    the argmax routes on -- not a magnitude/mean-abs proxy for it (the V3-EXQ-643
    same-statistic rule).

THE LIVE CHANNEL, AND WHY THE OCCUPANCY LIMIT DOES NOT KILL THIS RUN
-------------------------------------------------------------------
MECH-048's OTHER half (switching pressure) is NOT measurable on this substrate: the
contested modes are never occupied -- V3-EXQ-464b/464c/467d/464d ALL report
fraction_in_external_task = 0.0 at every seed with use_external_task_drive=True across
a 20x hysteresis sweep, mean_dwell is an episode-length artefact, and
n_switches == n_episodes. That is tracked separately as substrate_queue row
sd_salience_contested_mode_occupancy and is NOT in scope here.

This run is scoped to the ENTROPY half, and it is safe from that limit for a specific
structural reason: write_gate reads the SOFT operating_mode vector, not the discrete
current_mode. Temperature therefore modulates the gate CONTINUOUSLY even at zero
switches -- as p flattens toward uniform the gate is pulled toward the weight-mean of
its row, and as p sharpens the gate approaches the argmax mode's weight. For
e3_policy the row is {external_task 1.0, internal_planning 0.5, internal_replay 0.05,
offline_consolidation 0.3}: NON-uniform (a uniform row would make the gate invariant to
p and re-introduce vacuity by the other route). No occupancy-dependent DV is used.

CONSUMER FLAGS ARE NOT OPTIONAL. salience_apply_to_dacc_bias defaults to False, so the
e3_policy gate is inert unless it is switched on. A run with every consumer off would be
a FOREGONE NULL -- a structurally vacuous arm in the V3-EXQ-785 sense. It is switched on
here alongside use_dacc, and P3 measures that the channel is live.

WHY DIAGNOSTIC AND NOT EVIDENCE
-------------------------------
MECH-048 is currently exp=0 / lit=4 with a re-derive brake count of 0, so this would be
its FIRST experimental entry. Two of the three ways this run can end are substrate
readouts, not claim verdicts: (a) the environment fails to move mu (P1) -- says nothing
about MECH-048; (b) mu moves and entropy moves but no behavioural consequence appears --
that is a statement about V3's mode-prior CONSUMERS, not about whether opponent
stability overlays modulate mode-prior sharpness in the brain. Only (c) is
claim-relevant, and a diagnostic that routes to a properly-powered evidence run is the
honest way to reach it. diagnostic purpose is excluded from governance confidence /
conflict scoring, so a null here cannot corrupt the claim.

DESIGN
------
2 x 2 within-seed PAIRED factorial, 5 seeds:

    ENV REGIME (drives mu upstream)   x   COUPLING (the amendment)
      STABLE   resource-rich, low hazard,     OFF  use_stability_temperature=False
               periodic rest                  ON   use_stability_temperature=True,
      VOLATILE resource-sparse, high hazard,       mu_alpha=1.0, kappa_alpha=0.0
               no rest

Per seed the scaffolded_sd054_onboarding curriculum runs ONCE and the onboarded agent is
deep-copied into all four arms, so the arms differ ONLY in the two manipulated factors
and the DiD is within-subject. This is 4x cheaper than four independent onboardings and
a strictly stronger test of an interaction.

That sharing is exactly why every cell is emitted arm-reuse-INELIGIBLE
(extra_ineligible_reasons=["shared_onboarded_agent_across_arms"]). Under the
mint-as-you-go rule ineligibility is reserved for cells that genuinely share mutable
cross-cell state -- these do, so this is the required correctness guard and NOT a
"probably one-off" default.

PRE-REGISTERED CRITERIA (thresholds are constants below; nothing is derived post-hoc)
------------------------------------------------------------------------------------
C1 (LOAD-BEARING) visitation-entropy interaction exceeds max(absolute floor,
    MARGIN_SD x SD of the per-seed interaction).
C2 (corroborating, not load-bearing) same DiD form on committed-action-class entropy
    over FRESH selections only.

Both C1 and C2 are computed per seed (paired) and then aggregated, so the SD is the SD
of the DELTA -- an effect-size gate, not a bare threshold on a mean.

MEASUREMENT INTEGRITY -- the E3 latch
-------------------------------------
select_action returns the HELD action at agent.py:5458 BEFORE the salience block at
:5676, so the SalienceCoordinator ticks only on E3 ticks -- ~1 env step in 10
(heartbeat.e3_steps_per_tick defaults to 10). Every coordinator/PCC readout here is
gated through experiments/_lib/fresh_select.py (sentinel-key probe), and
n_latched_ticks / fresh_select_yield are emitted so the true denominator is auditable.
An ungated per-env-step read would pseudo-replicate ~10x AND would do so UNEQUALLY
across arms, because an arm that changes selection dynamics changes hold duration --
which is precisely the hold-weighted-readout defect class.

NOTE: commitment config does NOT exculpate this. The skip is the E3 CADENCE, not
beta_gate.is_elevated.

MECH-094: not applicable (waking arithmetic; no replay content authored).
Phased training: not applicable -- this script trains no head on any encoder output.
The coordinator overlay is non-trainable arithmetic with no gradient flow, and the
measurement phase runs under torch.no_grad().

Spec: REE_assembly/docs/architecture/sd_032_cingulate_integration_substrate.md
      ("Amendment (2026-07-21)"). See also ree-v3/CLAUDE.md "SD-032d AMENDMENT".
Contracts: tests/contracts/test_mech048_stability_temperature.py assert the WIRING,
not the science, and are NOT evidence for the claim.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
for _p in (str(_REPO), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiment_protocol import emit_outcome  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from scaffolded_sd054_onboarding import (  # noqa: E402
    ScaffoldedSD054OnboardingConfig,
    ScaffoldedSD054OnboardingScheduler,
    _build_env,
    _sense_with_optional_harm,
    stage_plan,
)
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.entropy_headroom import per_arm_headroom  # noqa: E402
from experiments._lib.manifest_core import stamp_recording_core  # noqa: E402
from experiments._lib.fresh_select import (  # noqa: E402
    FreshSelectCounter,
    FreshSelectProbe,
)
from experiments._lib.precondition_gate import (  # noqa: E402
    PreconditionSpec,
    aggregate_arm_gates,
    assert_no_structurally_unsatisfiable_gate,
    evaluate_arm_gate,
)
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_799_mech048_stability_temperature_behavioural_did"
EXPERIMENT_PURPOSE = "diagnostic"
CLAIM_IDS = ["MECH-048"]
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

EVIDENCE_DIR = (
    _REPO.parent / "REE_assembly" / "evidence" / "experiments"
)

FRESH_SELECT_NAMESPACE = "exq799"

SEEDS = [42, 43, 44, 45, 46]

# -- Arms -------------------------------------------------------------------
ENV_STABLE = "STABLE"
ENV_VOLATILE = "VOLATILE"
COUPLING_OFF = "OFF"
COUPLING_ON = "ON"

ARMS = [
    (ENV_STABLE, COUPLING_OFF),
    (ENV_STABLE, COUPLING_ON),
    (ENV_VOLATILE, COUPLING_OFF),
    (ENV_VOLATILE, COUPLING_ON),
]


def arm_id(env_regime: str, coupling: str) -> str:
    return env_regime + "_" + coupling


# -- Onboarding budgets (mirror the proven 797 / 603n curriculum) -----------
STAGE0_BUDGET = 20
STAGE0B_BUDGET = 10
P0_BUDGET = 100
HAZARD_STAGE_BUDGET = 40
P1_BUDGET = 50
P2_BUDGET = 15
TRAIN_STEPS = 200

ONBOARD_EPISODES = (
    STAGE0_BUDGET + STAGE0B_BUDGET + P0_BUDGET
    + HAZARD_STAGE_BUDGET + P1_BUDGET + P2_BUDGET
)

# -- Measurement phase ------------------------------------------------------
MEASURE_EPISODES = 20
MEASURE_STEPS = 200

# Total training-equivalent episodes per seed (progress denominator).
EPISODES_PER_RUN = ONBOARD_EPISODES + len(ARMS) * MEASURE_EPISODES

# -- Environment regime parameters (the UPSTREAM manipulation) --------------
# STABLE: resources plentiful, hazards few -> benefit events common, harm rare
#         -> success EMA rises. Plus periodic rest -> offline_recency reset.
# VOLATILE: resources scarce, hazards dense -> harm events common
#         -> success EMA falls. No rest -> offline_recency saturates.
# Neither writes pcc_stability. Both are ordinary CausalGridWorldV2 configs.
WORLD_DIM = 32
HARM_A_DIM = 16
HARM_OBS_A_DIM = 7
HARM_HISTORY_LEN = 10
ENV_SIZE = 12
STABLE_NUM_RESOURCES = 8
STABLE_NUM_HAZARDS = 1
STABLE_PROXIMITY_HARM = 0.05
STABLE_HFA = 0.0
# VOLATILE is deliberately SURVIVABLE. Authoring calibration (random-policy probe,
# 8-10 episodes per cell) measured the originally-drafted harsh regime
# (2 res / 6 haz / 0.35 / 0.6) at a mean episode length of 2-3 steps against
# STABLE's 38 -- the agent simply died. That is disqualifying twice over: there is
# no trajectory left to measure, and the episode-length asymmetry would depress
# visitation entropy MECHANICALLY, manufacturing a DV difference that has nothing
# to do with the mode prior. Hazard pressure is therefore kept mild and the
# manipulation carried by RESOURCE SCARCITY (task success / coherence) plus the
# rest schedule below. Measured mean episode length: 32 (VOLATILE) vs 38 (STABLE).
VOLATILE_NUM_RESOURCES = 2
VOLATILE_NUM_HAZARDS = 2
VOLATILE_PROXIMITY_HARM = 0.10
VOLATILE_HFA = 0.3

# dACC score-bias activation. See the block in _make_config for why
# dacc_foraging_weight stays at 0.0 (broadcast scalar = the 604c vacuity shape).
DACC_WEIGHT = 1.0
DACC_INTERACTION_WEIGHT = 0.5
DACC_BIAS_MAX_ABS = 1.0

# PCC weighting (see the rationale block in _make_config). Substrate defaults are
# fatigue 0.5 / offline 0.3 / window 500, under which a saturated drive_level pins
# mu at the [0,1] floor and no environmental manipulation can move it.
PCC_FATIGUE_WEIGHT = 0.15
PCC_OFFLINE_WEIGHT = 0.35
PCC_OFFLINE_RECENCY_WINDOW = 200

# Rest cadence in the STABLE regime (steps between enter_offline_mode calls).
# PCCConfig.offline_recency_window defaults to 500, so resting every 50 steps
# holds offline_recency near 0.1 while the VOLATILE arm saturates it at 1.0.
STABLE_REST_INTERVAL = 50

# Task-outcome encoding fed to agent.note_task_outcome(). Never a hand-set mu.
#
# Read from per-step DELTAS of info["total_benefit"] / info["total_harm"], NOT from
# the step's harm_signal return. The authoring smoke showed why: harm_signal is a
# CONTINUOUS proximity-harm reading that is negative on ~72% of STABLE steps and
# 100% of VOLATILE steps, so an `hs < 0 -> failure` encoding reports the safe
# regime as a near-constant stream of failures and INVERTS the manipulation
# (measured mu_separation = -0.013, i.e. the wrong sign).
#
# The delta form separates a genuine task event from that drizzle. Calibrated on a
# random-policy probe: in STABLE, benefit deltas fire on 7-19% of steps and harm
# deltas above HARM_EVENT_DELTA on ~5%, so BOTH outcome classes clear the 10%-ish
# coverage bar rather than one class saturating.
HARM_EVENT_DELTA = 0.1  # p90 of the continuous drizzle is 0.025; a real contact is >> this
OUTCOME_BENEFIT = 1.0
OUTCOME_HARM = 0.0
OUTCOME_NEUTRAL = 0.5

# -- Pre-registered thresholds ---------------------------------------------
# P1: the manipulation check. mu must actually separate across env regimes.
MU_SEPARATION_FLOOR = 0.15
# P2: mu must be UNAFFECTED by the coupling (it is upstream of it). Ceiling.
MU_COUPLING_INVARIANCE_CEIL = 0.02
# P3: the dACC bias must vary ACROSS CANDIDATES -- the same statistic the
# argmax routes on. A uniform bias is a broadcast constant and would be vacuous.
DACC_BIAS_RANGE_FLOOR = 1e-6
# P4: mechanism confirmation, coupling-ON arms only. This IS the arithmetic
# identity and is explicitly NOT load-bearing.
MODE_ENTROPY_MU_RESPONSE_FLOOR = 1e-4
# P5: enough genuine E3 selections for the DVs to have a real denominator.
FRESH_SELECT_FLOOR = 100

# C1 / C2 effect-size gates: scale on the SD of the per-seed DELTA, plus an
# absolute floor so a vanishing SD cannot manufacture a PASS.
MARGIN_SD = 1.0
C1_ABS_FLOOR = 0.02
C2_ABS_FLOOR = 0.02

# Non-gating entropy-headroom band (reported for every arm, never gates).
E_SAT_LOW = 0.02
E_SAT_HIGH = 0.98


def _device() -> torch.device:
    return torch.device("cpu")


def _shannon_entropy(counts: Counter) -> float:
    """Natural-log Shannon entropy of a count distribution."""
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        if c <= 0:
            continue
        p = float(c) / float(total)
        h -= p * math.log(p)
    return float(h)


def _normalised_entropy(counts: Counter, n_support: int) -> float:
    """Entropy normalised to [0, 1] by log(n_support). 0 when degenerate."""
    if n_support <= 1:
        return 0.0
    return float(_shannon_entropy(counts) / math.log(float(n_support)))


def _make_scaffold_cfg(dry_run: bool) -> ScaffoldedSD054OnboardingConfig:
    if dry_run:
        stage0, stage0b, p0, hazard, p1, p2, steps = 2, 2, 4, 3, 3, 2, 20
    else:
        stage0, stage0b, p0, hazard, p1, p2, steps = (
            STAGE0_BUDGET, STAGE0B_BUDGET, P0_BUDGET,
            HAZARD_STAGE_BUDGET, P1_BUDGET, P2_BUDGET, TRAIN_STEPS,
        )
    return ScaffoldedSD054OnboardingConfig(
        use_scaffolded_sd054_onboarding_scheduler=True,
        scaffold_stage0_enabled=True,
        scaffold_stage0_episode_budget=stage0,
        scaffold_stage0b_enabled=True,
        scaffold_stage0b_episode_budget=stage0b,
        scaffold_p0_episode_budget=p0,
        scaffold_hazard_stage_enabled=True,
        scaffold_hazard_stage_episode_budget=hazard,
        scaffold_p1_episode_budget=p1,
        scaffold_p2_episode_budget=p2,
        scaffold_steps_per_episode=steps,
        scaffold_developmental_window_enabled=True,
        scaffold_contact_gated_goal_updates=True,
        scaffold_feed_harm_stream=True,
    )


def _make_config(env) -> REEConfig:
    """Config common to ALL arms.

    The coupling factor is NOT set here -- it is flipped per arm on the live
    coordinator (agent.salience.config.use_stability_temperature) after the
    shared onboarding, so the arms are otherwise bit-identical.
    """
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=WORLD_DIM,
        alpha_world=0.9,
        use_harm_stream=True,
        use_affective_harm_stream=True,
        # Measured off the curriculum's own p2 env (harm_obs_a is 7-wide); the
        # REEConfig defaults are 50 and silently mis-size the affective-harm
        # encoder. Matches the 797 lineage.
        z_harm_a_dim=HARM_A_DIM,
        harm_obs_a_dim=HARM_OBS_A_DIM,
        harm_history_len=HARM_HISTORY_LEN,
        z_goal_enabled=True,
        # SD-032 stack. use_pcc_analog is the module under test's mu producer;
        # no experiment has ever exercised it (0 manifests mention it), which
        # is why P1 is a measured precondition rather than an assumption.
        use_dacc=True,
        use_salience_coordinator=True,
        use_pcc_analog=True,
        # THE dACC BIAS MUST BE SWITCHED ON, and this is not a detail.
        # DACCtoE3Adapter.forward's own docstring: "All multipliers default to 0,
        # so with default config the bias is the zero vector regardless of bundle
        # content." write_gate('e3_policy') multiplies that vector, so with the
        # defaults the gate scales ZERO and the temperature channel cannot reach
        # behaviour at all -- a guaranteed null. The authoring smoke measured
        # exactly that (dacc_bias_cross_candidate_range = 0.0 in all four arms).
        # dacc_weight activates the PER-CANDIDATE -mode_ev term; the interaction
        # term is per-candidate too.
        dacc_weight=DACC_WEIGHT,
        dacc_interaction_weight=DACC_INTERACTION_WEIGHT,
        # DELIBERATELY 0.0: foraging_value is a BROADCAST SCALAR that, in the
        # adapter's words, "uniformly raises all candidates' score ... without
        # biasing within the set". A broadcast constant cancels in the argmax --
        # it is the exact V3-EXQ-604c DV-symmetry vacuity shape, so it must not
        # be the term carrying this experiment's manipulation.
        dacc_foraging_weight=0.0,
        dacc_bias_max_abs=DACC_BIAS_MAX_ABS,
        # PCC weighting, re-scoped so mu HAS DYNAMIC RANGE in this env. With the
        # substrate defaults (fatigue 0.5 / offline 0.3 / window 500) the authoring
        # smoke measured drive_level saturated at 0.9, so the fatigue term alone
        # subtracts 0.45 from a 0.5 baseline and mu sits pinned at ~0.017 -- against
        # a [0,1] clip, with tau moving 1.7%. That is a STRUCTURALLY UNSATISFIABLE
        # precondition in the V3-EXQ-785 sense: no environmental manipulation could
        # clear the 0.15 separation floor, so the run could only ever self-route
        # substrate_not_ready_requeue.
        #
        # The remedy is to make the manipulation ABLE to reach the threshold, NOT to
        # lower the threshold (MU_SEPARATION_FLOOR is unchanged at 0.15 -- lowering
        # a gate to admit a pinned readout is exactly what turns a detected artifact
        # into a citable result). Fatigue is de-weighted so a saturated drive cannot
        # swamp the scalar, and the offline channel -- the one the manipulation
        # actually controls, via the rest schedule -- is given the larger share with
        # a window short enough to saturate inside one arm's step budget.
        # Predicted: STABLE ~0.32, VOLATILE ~0.02 (separation ~0.30, neither clipped).
        pcc_fatigue_weight=PCC_FATIGUE_WEIGHT,
        pcc_offline_weight=PCC_OFFLINE_WEIGHT,
        pcc_offline_recency_window=PCC_OFFLINE_RECENCY_WINDOW,
        # THE LIVE BEHAVIOURAL CHANNEL. Defaults to False, in which case
        # write_gate("e3_policy") is inert and this run would be a foregone
        # null (a structurally vacuous arm in the V3-EXQ-785 sense).
        salience_apply_to_dacc_bias=True,
        # Coupling knobs. use_stability_temperature stays OFF in the base
        # config; the ON arms flip it on the live coordinator. mu_alpha /
        # kappa_alpha are set to the landed defaults so that flipping the
        # master switch isolates the mu leg (kappa_alpha = 0.0).
        salience_use_stability_temperature=False,
        salience_temperature_mu_alpha=1.0,
        salience_temperature_kappa_alpha=0.0,
        salience_temperature_exponent_clip=4.0,
    )
    cfg.latent.use_resource_encoder = True
    return cfg


def _make_env(
    scaffold_cfg: ScaffoldedSD054OnboardingConfig,
    env_regime: str,
    seed: Optional[int] = None,
) -> CausalGridWorldV2:
    """Build the measurement env for one regime.

    Routed through the scaffold's own _build_env(phase="p2") rather than
    constructing CausalGridWorldV2 directly. That is load-bearing: the curriculum's
    envs enable limb_damage / reef / bipartite layout / SD-049 cue-recall, which
    ADD body- and world-observation dimensions. An env built by hand omits them and
    the observation no longer matches the encoder the agent was onboarded with
    (caught by the authoring smoke: body_obs 17 vs a 12-wide encoder).

    Going through _build_env also keeps the manipulation honest: STABLE and VOLATILE
    differ ONLY in resource density, hazard density, proximity harm and
    hazard-food-attraction. Every structural feature flag is identical, so the
    regimes are the same world at different difficulty -- not two different worlds.
    Nothing here touches the coordinator, the PCC, or pcc_stability.
    """
    cfg = copy.deepcopy(scaffold_cfg)
    if env_regime == ENV_STABLE:
        cfg.scaffold_p2_num_resources = STABLE_NUM_RESOURCES
        cfg.scaffold_p2_num_hazards = STABLE_NUM_HAZARDS
        cfg.scaffold_p2_proximity_harm_scale = STABLE_PROXIMITY_HARM
        cfg.scaffold_p2_hazard_food_attraction_guard = STABLE_HFA
    else:
        cfg.scaffold_p2_num_resources = VOLATILE_NUM_RESOURCES
        cfg.scaffold_p2_num_hazards = VOLATILE_NUM_HAZARDS
        cfg.scaffold_p2_proximity_harm_scale = VOLATILE_PROXIMITY_HARM
        cfg.scaffold_p2_hazard_food_attraction_guard = VOLATILE_HFA
    return _build_env(cfg, phase="p2", seed=seed)


def _onboard_seed(seed: int, scaffold_cfg, device) -> REEAgent:
    """Run the shared onboarding curriculum ONCE for this seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    probe_env = _make_env(scaffold_cfg, ENV_STABLE, seed=seed)
    probe_env.reset()
    cfg = _make_config(probe_env)
    agent = REEAgent(cfg).to(device)

    scheduler = ScaffoldedSD054OnboardingScheduler(scaffold_cfg)
    stages = {}
    print("Seed %d Condition onboarding" % seed, flush=True)
    for name, fn in (
        ("stage0", scheduler.run_stage0_nursery),
        ("stage0b", scheduler.run_stage0b_consolidation),
        ("p0", scheduler.run_p0),
        ("hazard", scheduler.run_hazard_avoidance),
        ("p1", scheduler.run_p1),
        ("p2", scheduler.run_p2),
    ):
        try:
            stages[name] = fn(agent, device)
        except TypeError:
            stages[name] = fn(agent, device=device)
        print(
            "  [train] onboarding seed=%d stage=%s ep %d/%d"
            % (seed, name, ONBOARD_EPISODES, EPISODES_PER_RUN),
            flush=True,
        )
    agent._exq799_onboarding_stages = stages
    return agent


def _run_measurement_arm(
    base_agent: REEAgent,
    scaffold_cfg: ScaffoldedSD054OnboardingConfig,
    env_regime: str,
    coupling: str,
    seed: int,
    device,
    n_eps: int,
    steps_per_ep: int,
    ep_offset: int,
) -> Dict[str, Any]:
    """Run one (env_regime x coupling) cell from a copy of the onboarded agent."""
    aid = arm_id(env_regime, coupling)

    agent = copy.deepcopy(base_agent)
    # Flip ONLY the coupling factor. Everything else is bit-identical to the
    # other arms of this seed -- that is what makes the DiD within-subject.
    if agent.salience is not None:
        agent.salience.config.use_stability_temperature = (coupling == COUPLING_ON)

    env = _make_env(scaffold_cfg, env_regime, seed=seed)

    probe = FreshSelectProbe(FRESH_SELECT_NAMESPACE)
    counter = FreshSelectCounter()

    visit_counts: Counter = Counter()
    action_counts: Counter = Counter()
    mu_vals: List[float] = []
    mode_entropy_vals: List[float] = []
    eff_temp_vals: List[float] = []
    e3_gate_vals: List[float] = []
    dacc_bias_ranges: List[float] = []
    success_ema_vals: List[float] = []
    offline_recency_vals: List[float] = []
    drive_vals: List[float] = []

    n_benefit = 0
    n_harm = 0
    n_neutral = 0
    n_rest = 0
    total_steps = 0
    n_episodes = 0
    prev_benefit: Optional[float] = None
    prev_harm: Optional[float] = None

    print("Seed %d Condition %s" % (seed, aid), flush=True)

    # FIXED STEP BUDGET, not a fixed episode count. Episodes end early in the
    # harsher regime, so an episode-count loop would give the two regimes
    # different numbers of steps -- and visitation entropy would then differ
    # MECHANICALLY (fewer steps can only visit fewer cells), manufacturing a DV
    # difference with no bearing on the mode prior. Consuming an identical step
    # budget in every arm removes that confound by construction; the episode
    # count becomes a reported diagnostic instead of a hidden denominator.
    step_budget = n_eps * steps_per_ep

    # CROSS-EPISODE, deliberately. Resetting this per episode was a real bug caught
    # by the authoring smoke: episodes run ~24 steps against a 50-step rest
    # interval, so the counter never reached the interval and n_rest_events was 0
    # in EVERY arm -- the primary mu channel was silently dead. It also mirrors the
    # substrate: PCCAnalog.reset() deliberately does NOT reset _steps_since_offline,
    # because "the agent did not just rest because a new episode began".
    steps_since_rest = 0

    with torch.no_grad():
        while total_steps < step_budget:
            _, obs_dict = env.reset()
            agent.reset()
            n_episodes += 1
            prev_benefit = None
            prev_harm = None

            while total_steps < step_budget:
                obs_body = obs_dict["body_state"].to(device)
                obs_world = obs_dict["world_state"].to(device)
                latent = _sense_with_optional_harm(
                    agent, obs_body, obs_world, obs_dict, device, True
                )

                ticks = agent.clock.advance()
                e1_prior = (
                    agent._e1_tick(latent) if ticks.get("e1_tick")
                    else torch.zeros(1, WORLD_DIM, device=device)
                )
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)

                with probe.watch(agent) as fresh:
                    action = agent.select_action(candidates, ticks)
                is_fresh = bool(fresh)
                counter.record(is_fresh)

                action_idx = int(action.argmax(dim=-1).item())
                total_steps += 1

                # Behavioural DV 1: state visitation over the realized
                # trajectory. Recorded EVERY env step -- it is a property of
                # where the body went, not of a latched E3 readout, so the
                # latch does not apply to it.
                visit_counts[(int(env.agent_x), int(env.agent_y))] += 1

                # Coordinator / PCC readouts: FRESH E3 ticks ONLY. On a latched
                # tick these attributes still hold the previous selection's
                # values, so recording them would pseudo-replicate ~10x and do
                # so unequally across arms.
                if is_fresh:
                    # Behavioural DV 2: committed action class, counted once
                    # per genuine selection rather than once per env step.
                    action_counts[action_idx] += 1

                    last = agent._salience_last_tick
                    if isinstance(last, dict):
                        mode_entropy_vals.append(float(last.get("mode_entropy", 0.0)))
                        eff_temp_vals.append(
                            float(last.get("effective_temperature", 0.0))
                        )
                    pcc_tick = getattr(agent, "_pcc_last_tick", None)
                    if isinstance(pcc_tick, dict):
                        mu_vals.append(float(pcc_tick.get("pcc_stability", 0.0)))
                        success_ema_vals.append(
                            float(pcc_tick.get("success_ema", 0.0))
                        )
                        offline_recency_vals.append(
                            float(pcc_tick.get("offline_recency", 0.0))
                        )
                        drive_vals.append(float(pcc_tick.get("drive_level", 0.0)))
                    if agent.salience is not None:
                        e3_gate_vals.append(
                            float(agent.salience.write_gate("e3_policy"))
                        )
                    # P3 positive control: the CROSS-CANDIDATE RANGE of the
                    # dACC score bias -- the same statistic the argmax routes
                    # on. A uniform bias is a broadcast constant (vacuous).
                    bias = getattr(agent, "_dacc_last_bias", None)
                    if bias is not None:
                        try:
                            b = bias.detach().reshape(-1)
                            if b.numel() > 1:
                                dacc_bias_ranges.append(
                                    float(b.max().item() - b.min().item())
                                )
                        except Exception:
                            pass

                _, _harm_signal, done, info, obs_dict = env.step(action_idx)

                # UPSTREAM mu drive #1: task outcome, read from env event DELTAS
                # (see HARM_EVENT_DELTA for why not harm_signal). This is the ONLY
                # thing fed into the success EMA; pcc_stability is never written.
                cur_benefit = float(info.get("total_benefit", 0.0))
                cur_harm = float(info.get("total_harm", 0.0))
                if prev_benefit is None:
                    outcome = OUTCOME_NEUTRAL
                else:
                    d_benefit = cur_benefit - prev_benefit
                    d_harm = cur_harm - prev_harm
                    if d_benefit > 0.0:
                        outcome = OUTCOME_BENEFIT
                        n_benefit += 1
                    elif d_harm > HARM_EVENT_DELTA:
                        outcome = OUTCOME_HARM
                        n_harm += 1
                    else:
                        outcome = OUTCOME_NEUTRAL
                        n_neutral += 1
                prev_benefit, prev_harm = cur_benefit, cur_harm
                agent.note_task_outcome(outcome)

                # UPSTREAM mu drive #2: rest schedule. STABLE rests
                # periodically (offline_recency near 0); VOLATILE never rests
                # (offline_recency saturates at 1.0).
                steps_since_rest += 1
                if (
                    env_regime == ENV_STABLE
                    and steps_since_rest >= STABLE_REST_INTERVAL
                ):
                    agent.enter_offline_mode()
                    steps_since_rest = 0
                    n_rest += 1

                if done:
                    break

            counter.flush()
            # Progress is reported against the STEP budget (converted to episode
            # units) so the denominator still matches episodes_per_run even though
            # the arm is step-budgeted rather than episode-budgeted.
            eps_done = min(n_eps, int(total_steps / max(1, steps_per_ep)))
            print(
                "  [train] measure seed=%d arm=%s ep %d/%d"
                % (seed, aid, ep_offset + eps_done, EPISODES_PER_RUN),
                flush=True,
            )

    counter.flush()

    n_cells = ENV_SIZE * ENV_SIZE
    n_actions = int(getattr(env, "action_dim", 0)) or (
        max(action_counts) + 1 if action_counts else 1
    )

    def _mean(vals: List[float]) -> float:
        return float(statistics.fmean(vals)) if vals else 0.0

    return {
        "arm": aid,
        "env_regime": env_regime,
        "coupling": coupling,
        "seed": seed,
        # -- Load-bearing behavioural DVs --
        "visitation_entropy_norm": _normalised_entropy(visit_counts, n_cells),
        "action_class_entropy_norm": _normalised_entropy(action_counts, n_actions),
        "n_distinct_cells": len(visit_counts),
        # -- Mechanism readouts (NOT load-bearing: mu -> tau -> H is an identity) --
        "pcc_stability_mean": _mean(mu_vals),
        "mode_entropy_mean": _mean(mode_entropy_vals),
        "effective_temperature_mean": _mean(eff_temp_vals),
        "e3_policy_gate_mean": _mean(e3_gate_vals),
        "dacc_bias_cross_candidate_range_mean": _mean(dacc_bias_ranges),
        # -- Upstream mu components, so a P1 failure is attributable --
        "success_ema_mean": _mean(success_ema_vals),
        "offline_recency_mean": _mean(offline_recency_vals),
        "drive_level_mean": _mean(drive_vals),
        # -- Event / denominator audit --
        "n_benefit_events": n_benefit,
        "n_harm_events": n_harm,
        "n_neutral_steps": n_neutral,
        "n_rest_events": n_rest,
        # Outcome-class coverage: BOTH classes must be populated or the success
        # EMA is driven by one label and the manipulation is degenerate.
        "benefit_outcome_frac": (
            float(n_benefit) / float(total_steps) if total_steps else 0.0
        ),
        "harm_outcome_frac": (
            float(n_harm) / float(total_steps) if total_steps else 0.0
        ),
        "n_episodes": n_episodes,
        "mean_episode_len": (
            float(total_steps) / float(n_episodes) if n_episodes else 0.0
        ),
        "n_env_steps": total_steps,
        "n_fresh_select": counter.n_fresh_select,
        "n_latched_ticks": counter.n_latched,
        "fresh_select_yield": counter.fresh_select_yield(total_steps),
        "replication_factor_avoided": counter.replication_factor(total_steps),
        "hold_duration_mean": counter.hold_duration_mean(),
    }


def _run_seed(
    seed: int, scaffold_cfg, device, n_eps: int, steps_per_ep: int
) -> List[Dict[str, Any]]:
    """Shared onboarding, then all four arms from copies of that agent."""
    base_agent = _onboard_seed(seed, scaffold_cfg, device)
    base_agent.eval()

    rows: List[Dict[str, Any]] = []
    for i, (env_regime, coupling) in enumerate(ARMS):
        full_cfg = {
            "env_regime": env_regime,
            "coupling": coupling,
            "env_size": ENV_SIZE,
            "measure_episodes": n_eps,
            "measure_steps": steps_per_ep,
            "onboard_episodes": ONBOARD_EPISODES,
        }
        with arm_cell(
            seed,
            config_slice=full_cfg,
            script_path=Path(__file__),
            # The four arms of a seed SHARE one onboarded agent, so the cells
            # are not independent. Reuse-ineligible is the required
            # correctness guard here, not a mint-as-you-go opt-out.
            extra_ineligible_reasons=["shared_onboarded_agent_across_arms"],
        ) as cell:
            row = _run_measurement_arm(
                base_agent, scaffold_cfg, env_regime, coupling, seed, device,
                n_eps, steps_per_ep,
                ep_offset=ONBOARD_EPISODES + i * n_eps,
            )
            cell.stamp(row)
        rows.append(row)
    return rows


# -- Preconditions -----------------------------------------------------------

PRECONDITION_SPECS = [
    PreconditionSpec(
        name="mu_separation_env",
        description=(
            "The ENVIRONMENT actually moves mu: mean pcc_stability(STABLE) - "
            "mean pcc_stability(VOLATILE), measured on fresh E3 ticks. This is "
            "THE manipulation check -- use_pcc_analog has never been exercised "
            "by any experiment (0 manifests), so it is measured, never assumed."
        ),
        control=(
            "STABLE (resource-rich, low hazard, periodic rest) vs VOLATILE "
            "(resource-sparse, hazard-dense, no rest); outcomes fed from the "
            "env harm_signal. pcc_stability is never written directly."
        ),
        threshold=MU_SEPARATION_FLOOR,
        direction="lower",
        kind="readiness",
    ),
    PreconditionSpec(
        name="mu_invariant_to_coupling",
        description=(
            "mu is UPSTREAM of the coupling, so it must not differ between "
            "coupling ON and OFF within an env regime. A violation means the "
            "coupling is feeding back into its own driver and the DiD contrast "
            "is confounded."
        ),
        control="max over env regimes of |mu(ON) - mu(OFF)| within regime.",
        threshold=MU_COUPLING_INVARIANCE_CEIL,
        direction="upper",
        kind="readiness",
    ),
    PreconditionSpec(
        name="dacc_bias_cross_candidate_range",
        description=(
            "The dACC score bias must vary ACROSS CANDIDATES. write_gate"
            "('e3_policy') multiplies that vector, so a bias uniform across "
            "candidates is a broadcast constant that cancels in the argmax -- "
            "the 604c DV-symmetry vacuity. This asserts the SAME statistic the "
            "load-bearing criterion routes on (range), NOT a magnitude proxy."
        ),
        control=(
            "Fresh E3 ticks with use_dacc=True and "
            "salience_apply_to_dacc_bias=True, i.e. the live behavioural channel."
        ),
        threshold=DACC_BIAS_RANGE_FLOOR,
        direction="lower",
        kind="readiness",
    ),
    PreconditionSpec(
        name="mode_entropy_responds_to_mu",
        description=(
            "MECHANISM CONFIRMATION ONLY, and explicitly NOT load-bearing: with "
            "the coupling ON, H(operating_mode) must differ across env regimes "
            "by more than the floor. This is the ARITHMETIC IDENTITY of the "
            "temperature coupling -- it is recorded to show the overlay fired, "
            "and must never be cited as evidence for MECH-048."
        ),
        control="coupling-ON arms only; |H(STABLE,ON) - H(VOLATILE,ON)|.",
        threshold=MODE_ENTROPY_MU_RESPONSE_FLOOR,
        direction="lower",
        kind="mechanism",
        # Scoped OUT of the coupling-OFF arms: with the switch off the
        # temperature is softmax_temperature exactly, so demanding an entropy
        # response there would be structurally unsatisfiable and would (785)
        # silently vacate the ON arms' valid result.
        applies_to=lambda ctx: ctx.get("coupling") == COUPLING_ON,
        applies_note=(
            "use_stability_temperature is False in this arm by construction, so "
            "mu cannot reach the mode prior and an entropy response is not "
            "meaningful here."
        ),
    ),
    PreconditionSpec(
        name="fresh_select_sufficiency",
        description=(
            "Enough genuine E3 selections for the action-entropy DV to have a "
            "real denominator. The E3 cadence is 10, so env steps are NOT "
            "selections."
        ),
        control="n_fresh_select for this arm, sentinel-key gated.",
        threshold=FRESH_SELECT_FLOOR,
        direction="lower",
        kind="readiness",
    ),
]


def _arm_context(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "arm_id": row["arm"],
        "env_regime": row["env_regime"],
        "coupling": row["coupling"],
    }


def _did(rows_by_arm: Dict[str, float]) -> float:
    """Difference-in-differences on one DV, for a single seed."""
    return (
        (rows_by_arm[arm_id(ENV_VOLATILE, COUPLING_ON)]
         - rows_by_arm[arm_id(ENV_STABLE, COUPLING_ON)])
        - (rows_by_arm[arm_id(ENV_VOLATILE, COUPLING_OFF)]
           - rows_by_arm[arm_id(ENV_STABLE, COUPLING_OFF)])
    )


def _effect_gate(deltas: List[float], abs_floor: float) -> Dict[str, Any]:
    """Effect-size gate: |mean| > max(abs_floor, MARGIN_SD * SD(delta))."""
    n = len(deltas)
    mean = float(statistics.fmean(deltas)) if n else 0.0
    sd = float(statistics.stdev(deltas)) if n > 1 else 0.0
    margin = max(abs_floor, MARGIN_SD * sd)
    return {
        "n_seeds": n,
        "mean": mean,
        "sd": sd,
        "margin": margin,
        "abs_floor": abs_floor,
        "passed": bool(abs(mean) > margin),
    }


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    t0 = time.perf_counter()
    device = _device()
    scaffold_cfg = _make_scaffold_cfg(dry_run)

    seeds = SEEDS[:2] if dry_run else SEEDS
    n_eps = 4 if dry_run else MEASURE_EPISODES
    steps_per_ep = 120 if dry_run else MEASURE_STEPS

    # Design-time structural check: refuse BEFORE spending compute if any
    # precondition is unsatisfiable from the pre-registered config.
    arm_contexts = [
        {"arm_id": arm_id(e, c), "env_regime": e, "coupling": c}
        for (e, c) in ARMS
    ]
    assert_no_structurally_unsatisfiable_gate(
        PRECONDITION_SPECS, arm_contexts, arm_id_key="arm_id"
    )

    all_rows: List[Dict[str, Any]] = []
    for seed in seeds:
        all_rows.extend(
            _run_seed(seed, scaffold_cfg, device, n_eps, steps_per_ep)
        )

    by_seed: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for r in all_rows:
        by_seed.setdefault(r["seed"], {})[r["arm"]] = r

    complete_seeds = [
        s for s, m in by_seed.items()
        if all(arm_id(e, c) in m for (e, c) in ARMS)
    ]
    complete_seeds.sort()

    # -- Per-arm precondition measurements --------------------------------
    def _arm_rows(aid: str) -> List[Dict[str, Any]]:
        return [r for r in all_rows if r["arm"] == aid]

    def _arm_mean(aid: str, key: str) -> float:
        vals = [r[key] for r in _arm_rows(aid)]
        return float(statistics.fmean(vals)) if vals else 0.0

    mu_stable = float(statistics.fmean([
        _arm_mean(arm_id(ENV_STABLE, c), "pcc_stability_mean")
        for c in (COUPLING_OFF, COUPLING_ON)
    ]))
    mu_volatile = float(statistics.fmean([
        _arm_mean(arm_id(ENV_VOLATILE, c), "pcc_stability_mean")
        for c in (COUPLING_OFF, COUPLING_ON)
    ]))
    mu_separation = mu_stable - mu_volatile

    mu_coupling_gap = max(
        abs(_arm_mean(arm_id(e, COUPLING_ON), "pcc_stability_mean")
            - _arm_mean(arm_id(e, COUPLING_OFF), "pcc_stability_mean"))
        for e in (ENV_STABLE, ENV_VOLATILE)
    )

    entropy_mu_response = abs(
        _arm_mean(arm_id(ENV_STABLE, COUPLING_ON), "mode_entropy_mean")
        - _arm_mean(arm_id(ENV_VOLATILE, COUPLING_ON), "mode_entropy_mean")
    )

    arm_gates = []
    for (e, c) in ARMS:
        aid = arm_id(e, c)
        ctx = {"arm_id": aid, "env_regime": e, "coupling": c}
        # Worst cell (not a mean) for the sufficiency floor, so `met`
        # recomputes exactly and names the offender.
        rows_a = _arm_rows(aid)
        worst_fresh = min((r["n_fresh_select"] for r in rows_a), default=0)
        measured = {
            "mu_separation_env": mu_separation,
            "mu_invariant_to_coupling": mu_coupling_gap,
            "dacc_bias_cross_candidate_range": _arm_mean(
                aid, "dacc_bias_cross_candidate_range_mean"
            ),
            "fresh_select_sufficiency": float(worst_fresh),
        }
        if c == COUPLING_ON:
            measured["mode_entropy_responds_to_mu"] = entropy_mu_response
        arm_gates.append(evaluate_arm_gate(aid, ctx, PRECONDITION_SPECS, measured))

    gate_agg = aggregate_arm_gates(arm_gates)

    # -- Criteria (per-seed paired DiD) ------------------------------------
    c1_deltas = [
        _did({a: by_seed[s][a]["visitation_entropy_norm"] for a in by_seed[s]})
        for s in complete_seeds
    ]
    c2_deltas = [
        _did({a: by_seed[s][a]["action_class_entropy_norm"] for a in by_seed[s]})
        for s in complete_seeds
    ]
    c1 = _effect_gate(c1_deltas, C1_ABS_FLOOR)
    c2 = _effect_gate(c2_deltas, C2_ABS_FLOOR)

    # ALL-GREEN, deliberately -- and this is NOT the V3-EXQ-785 whole-run AND defect.
    # 785 forbids ANDing because it lets one arm's structurally-impossible
    # precondition vacate ANOTHER arm's independent, well-powered result. Here there
    # is no per-arm result to preserve: the load-bearing DV is a single
    # difference-in-differences CONTRAST spanning all four arms, so it is
    # uninterpretable unless every arm in the contrast is readable. The full
    # per_arm_gate is still emitted, so a red arm stays individually attributable.
    gate_green = bool(gate_agg.get("all_green", False))
    overall_pass = bool(gate_green and c1["passed"])

    # -- Self-routing interpretation grid ----------------------------------
    failed = (gate_agg.get("per_arm_gate") or {}).get(
        "failed_preconditions_by_arm"
    ) or {}
    flat_failed = {
        n.split("::")[-1] for names in failed.values() for n in names
    }
    if not gate_green:
        if "mu_separation_env" in flat_failed:
            label = "substrate_not_ready_requeue"
            routing = (
                "The environmental manipulation did not move pcc_stability past "
                "the floor, so nothing downstream was ever exercised. This says "
                "NOTHING about MECH-048. Re-queue with a stronger upstream "
                "contrast (wider resource/hazard split, longer episodes so the "
                "success EMA can travel, or a shorter rest interval)."
            )
        elif "dacc_bias_cross_candidate_range" in flat_failed:
            label = "substrate_not_ready_requeue"
            routing = (
                "The dACC bias did not vary across candidates, so the e3_policy "
                "gate was a broadcast constant and the behavioural channel was "
                "arithmetically inert. Repair the instrument before any verdict."
            )
        elif "fresh_select_sufficiency" in flat_failed:
            label = "substrate_not_ready_requeue"
            routing = (
                "Too few genuine E3 selections for the action-entropy DV. "
                "Re-queue with more measurement episodes."
            )
        else:
            label = "substrate_not_ready_requeue"
            routing = "A readiness precondition was unmet; see per_arm_gate."
    elif overall_pass:
        label = "mu_overlay_has_behavioural_authority"
        routing = (
            "Environmentally-driven mu changed BEHAVIOUR through the mode-prior "
            "temperature, over and above every non-temperature env path (which "
            "the coupling-OFF arm carries). This is the precondition for a "
            "properly-powered EVIDENCE run against MECH-048 -- route to "
            "/queue-experiment for that, do NOT promote on this diagnostic."
        )
    else:
        label = "mu_overlay_entropy_only_no_behavioural_authority"
        routing = (
            "mu moved, the mode-prior entropy moved with it (mechanism "
            "confirmed), but no behavioural consequence survived the DiD. This "
            "is a statement about V3's mode-prior CONSUMERS, not a falsification "
            "of MECH-048: the entropy half is implemented and fires, but nothing "
            "downstream converts it into action at a detectable scale. Route to "
            "/implement-substrate on the consumer path (write_gate breadth), and "
            "note the interaction with the sd_salience_contested_mode_occupancy "
            "row -- with contested modes unoccupied the soft vector stays near "
            "one-hot, which bounds how much the gate can travel."
        )

    # Both criteria are DiD contrasts spanning EVERY arm, so neither belongs to a
    # single arm and the per-arm keying of arm_criteria_non_degenerate does not
    # apply. Degeneracy here is exactly "was any arm of the contrast unreadable".
    criteria_nd = {
        "C1_visitation_entropy_did": gate_green,
        "C2_action_entropy_did": gate_green,
    }

    headroom = per_arm_headroom(
        all_rows,
        value_key="mode_entropy_mean",
        low=E_SAT_LOW,
        high=E_SAT_HIGH,
    )

    manifest: Dict[str, Any] = {
        "run_id": "%s_%s_v3" % (
            EXPERIMENT_TYPE, datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        ),
        "experiment_type": EXPERIMENT_TYPE,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "claim_ids": CLAIM_IDS,
        "outcome": "PASS" if overall_pass else "FAIL",
        "timestamp_utc": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "sleep_driver_pattern": (
            "N/A (waking measurement phase; STABLE regime calls "
            "enter_offline_mode() as a REST event only)"
        ),
        "dry_run": bool(dry_run),
        "arm_results": all_rows,
        "per_seed_c1_did": dict(zip(map(str, complete_seeds), c1_deltas)),
        "per_seed_c2_did": dict(zip(map(str, complete_seeds), c2_deltas)),
        "criteria": [
            {
                "name": "C1_visitation_entropy_did",
                "load_bearing": True,
                "passed": bool(c1["passed"]),
                "detail": c1,
            },
            {
                "name": "C2_action_entropy_did",
                "load_bearing": False,
                "passed": bool(c2["passed"]),
                "detail": c2,
            },
        ],
        "per_arm_gate": gate_agg.get("per_arm_gate", gate_agg),
        "non_degenerate": gate_green,
        "degeneracy_reason": gate_agg.get("degeneracy_reason", ""),
        "interpretation": {
            "label": label,
            "routing_detail": routing,
            "preconditions": gate_agg.get(
                "adjudication_preconditions", gate_agg.get("preconditions", [])
            ),
            "preconditions_scope_note": gate_agg.get("preconditions_scope_note", ""),
            "criteria_non_degenerate": criteria_nd,
        },
        "diagnostics": {
            "entropy_headroom_per_arm": headroom,
            "mu_stable_mean": mu_stable,
            "mu_volatile_mean": mu_volatile,
            "mu_separation": mu_separation,
            "mu_coupling_gap": mu_coupling_gap,
            "mode_entropy_mu_response_on_arms": entropy_mu_response,
        },
        "vacuity_note": (
            "mu -> tau -> H(operating_mode) is an ARITHMETIC IDENTITY of the "
            "temperature coupling and is recorded as mechanism confirmation "
            "ONLY (precondition mode_entropy_responds_to_mu, kind=mechanism, "
            "never load_bearing). The load-bearing DV is the coupling-vs-control "
            "difference-in-differences on a trajectory-level behavioural "
            "readout, which no arithmetic fixes in advance."
        ),
        "stage_plan": stage_plan(),
    }

    full_config = {
        "seeds": seeds,
        "arms": [arm_id(e, c) for (e, c) in ARMS],
        "onboard_episodes": ONBOARD_EPISODES,
        "measure_episodes": n_eps,
        "measure_steps": steps_per_ep,
        "env_size": ENV_SIZE,
        "stable": {
            "num_resources": STABLE_NUM_RESOURCES,
            "num_hazards": STABLE_NUM_HAZARDS,
            "proximity_harm_scale": STABLE_PROXIMITY_HARM,
            "hazard_food_attraction": STABLE_HFA,
            "rest_interval": STABLE_REST_INTERVAL,
        },
        "volatile": {
            "num_resources": VOLATILE_NUM_RESOURCES,
            "num_hazards": VOLATILE_NUM_HAZARDS,
            "proximity_harm_scale": VOLATILE_PROXIMITY_HARM,
            "hazard_food_attraction": VOLATILE_HFA,
            "rest_interval": None,
        },
        "thresholds": {
            "mu_separation_floor": MU_SEPARATION_FLOOR,
            "mu_coupling_invariance_ceil": MU_COUPLING_INVARIANCE_CEIL,
            "dacc_bias_range_floor": DACC_BIAS_RANGE_FLOOR,
            "mode_entropy_mu_response_floor": MODE_ENTROPY_MU_RESPONSE_FLOOR,
            "fresh_select_floor": FRESH_SELECT_FLOOR,
            "margin_sd": MARGIN_SD,
            "c1_abs_floor": C1_ABS_FLOOR,
            "c2_abs_floor": C2_ABS_FLOOR,
        },
        "substrate_flags": {
            "use_salience_coordinator": True,
            "use_pcc_analog": True,
            "use_dacc": True,
            "salience_apply_to_dacc_bias": True,
            "salience_temperature_mu_alpha": 1.0,
            "salience_temperature_kappa_alpha": 0.0,
            "salience_temperature_exponent_clip": 4.0,
        },
    }

    # Multi-arm: stamp AFTER arm_results is assembled so substrate_hash HOISTS
    # from the per-cell fingerprints instead of being recomputed.
    stamp_recording_core(
        manifest,
        config=full_config,
        seeds=seeds,
        script_path=Path(__file__),
        started_at=t0,
    )

    for s in complete_seeds:
        print("verdict: %s" % ("PASS" if overall_pass else "FAIL"), flush=True)

    return manifest


def main(dry_run: bool = False) -> Dict[str, Any]:
    manifest = run_experiment(dry_run=dry_run)
    out_path = write_flat_manifest(
        manifest, EVIDENCE_DIR, dry_run=dry_run, stamp=False
    )
    print("outcome: %s" % manifest["outcome"], flush=True)
    print("label: %s" % manifest["interpretation"]["label"], flush=True)
    print("manifest: %s" % out_path, flush=True)
    return {"manifest": manifest, "out_path": out_path}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=EXPERIMENT_TYPE)
    parser.add_argument("--dry-run", action="store_true", help="short smoke run")
    args = parser.parse_args()

    res = main(dry_run=args.dry_run)
    _outcome_raw = str(res["manifest"]["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=res["out_path"],
        dry_run=args.dry_run,
    )
