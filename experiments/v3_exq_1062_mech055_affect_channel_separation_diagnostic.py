#!/opt/local/bin/python3
"""
V3-EXQ-1062: MECH-055 affective channel separation -- the NARROWED
two-axis-plus-harm-only REPRESENTATIONAL separation diagnostic.

experiment_purpose: diagnostic

MECH-055 asserts that hedonic tone, valence, and signed PE stay distinct
channels. Its own what_would_answer (claims.yaml, 2026-08-08) states that the
FULL three-axis-with-harm/benefit-duality test CANNOT yet run -- MECH-054's
2026-08-08 finding is that only the HARM side of axis 3 is real, with no
benefit-side forward model and no benefit-side precision tracker anywhere in
ree_core -- and that "a narrower two-axis-plus-harm-only test CAN run now".
This is that narrower test. It is deliberately scoped to the REPRESENTATIONAL
level, which the claim explicitly licenses: "downstream behavioural legibility
gap is separate and doesn't block a representational-level test."

WHAT THIS RUN DOES AND DOES NOT DECIDE
--------------------------------------
IN SCOPE. The claim's own two FALSIFIERS, at the representational level:
  (i)  "any two axes move in a fixed, predictable ratio under manipulations
       designed to perturb only one"                              -> C2
  (ii) "VALENCE_HARM_DISCRIMINATIVE and the harm-side forward-model PE are
       numerically redundant (correlation ~1) under a decoupling
       manipulation"                                              -> C1
plus the claim's second CONFIRMING clause, that the two harm representations
"show measurably different variance/timing signatures under a manipulation
that decouples their upstream sources"                            -> C3, C4

OUT OF SCOPE, stated so no reader over-reads a PASS:
  (a) The benefit-side signed-PE channel. It does not exist (MECH-054). No
      result here can verdict the harm/benefit duality the claim's wording
      requires, which is why a clean pass records evidence_direction "mixed"
      and NOT "supports", and why experiment_purpose is "diagnostic".
  (b) The claim's FIRST confirming clause in its behavioural form -- effects on
      "that axis's own documented downstream role". V3-EXQ-799 confirmed axis
      1's internal mechanism fires (mode-prior entropy moves with mu, 1.105
      nats) but found NO behavioural DV downstream with enough sensitivity to
      detect it (write_gate breadth gap, routed to /implement-substrate, not
      landed). This run therefore measures the axes' own values and their
      joint structure, not a behavioural consequence. It cannot re-derive the
      conversion / F-dominance ceiling because it has no behavioural DV.
  (c) MECH-035's ranking claim (Pareto/lexicographic vs scalar). The
      cross-candidate valence range is RECORDED here as a non-gating
      diagnostic, not tested -- MECH-035 is deliberately NOT tagged.

THE DECOUPLING MANIPULATION, AND THE THREE LEVERS RULED OUT FIRST
-----------------------------------------------------------------
The claim demands "a manipulation that decouples their upstream sources".
VALENCE_HARM_DISCRIMINATIVE is fed by z_harm_s (a LEVEL, RBF-smoothed onto map
nodes); the dACC channel is fed by z_harm_a through E2HarmAForward (a temporal
forward-model RESIDUAL, precision-weighted -- MECH-258). Three candidate
levers were checked against the substrate and REJECTED before this design:

  1. SD-021 / AIC descending attenuation of z_harm_s. REJECTED: the gain is
     harm_s_gain = 1 - base_attenuation * mode_weight * drive_protect, and
     mode_weight = p_external * (1.0 if beta_gate_elevated else 0.0)
     (ree_core/cingulate/aic_analog.py:249). It is gated on the COMMITMENT
     latch, so it cannot fire commitment-free.
  2. harm_nonredundancy_weight (the SD-019 cosine^2 penalty between z_harm_s
     and z_harm_a). REJECTED as a FLOOR: V3-EXQ-323 measured BASELINE
     cosine_sq at 8.5e-05 to 0.025 across seeds -- the streams are already
     near-orthogonal unpenalised, the penalty has no headroom, and 323 failed
     its own C1 in 2/5 seeds for exactly that reason. (Read positively, this
     is good news for MECH-055 and is why the LATENTS are not what C1 tests.)
     Note also that this knob is a dataclass field with NO from_dims kwarg, so
     passing it to from_dims is silently swallowed -- verified, not assumed.
  3. env_drift_interval / env_drift_prob (hazard relocation). REJECTED as a
     MEASURED NULL: causal_grid_world.py's own SD-MEL-PRODUCER note records
     that "the optimal prediction of a random walk is its mean", and
     V3-EXQ-677's env_drift_interval 999 -> 3 manipulation produced a
     high-vs-low mean-PE difference of 8.8e-07 against a 0.01 threshold.

ADOPTED: world_rule_shift (SD-MEL-PRODUCER), the lever built because drift
failed. It periodically re-permutes the action -> displacement map, so every
ACTION-CONDITIONED forward model -- E2HarmAForward included -- becomes
systematically wrong and stays wrong until re-learned. That raises the
prediction RESIDUAL while harm EXPOSURE stays matched (matched-ness is a
measured precondition here, never an assumption). V3-EXQ-861e confirmed the IV
is non-degenerate on this substrate: its
ecological_novelty_mel_gradient_present_this_config precondition and its
C1_measured_mel_gradient_present / C1_dv_spread_nonzero non-degeneracy flags
were all met (861e's FAIL was a downstream MEL-coupling DV, not the IV).

  ARM_0_STATIONARY  world_rule_shift off            (reference)
  ARM_2_HIGH_SHIFT  interval 10, depth 2

A third, intermediate rung (interval 25) was designed and DROPPED on measured
cost, not on doubt: this experiment's own Step 2.5a probe measured ~0.84 s of
CPU per env step on this config (32 candidates re-proposed on ~99% of steps),
and no criterion here consumes a middle rung -- C3 compares the stationary and
high arms, C2 and C4 are computed per arm. A graded dose-response successor is
the natural follow-on IF C3 passes; it is not a precondition for deciding
either of the claim's falsifiers.

env_drift is PINNED OFF in every arm (interval 999, prob 0.0, following 861e)
so the rule-shift ladder is the only world-nonstationarity that varies.

DV-SYMMETRY INVARIANCE -- one statement per arm (MANDATORY DECLARATION)
----------------------------------------------------------------------
Every DV here is a Spearman correlation, an OLS R^2, a lag-1 autocorrelation,
or a RELATIVE change -- computed over per-tick series. The symmetry group of
that family is: independent positive affine rescaling of either series, plus a
uniform additive constant on either series (correlations and R^2 are exactly
invariant under both; a relative change is invariant under rescaling only).

ARM_0_STATIONARY, ARM_1_MED_SHIFT, ARM_2_HIGH_SHIFT -- the same statement
holds for all three, because they differ only in the rate of one manipulation:
the manipulation is a re-permutation of the action -> displacement map, which
REORDERS and RE-CONTENTS the temporal sequence of z_harm_a and destroys its
action-conditioned predictability. That is neither an affine rescaling nor an
added constant of any measured series, so NO arm's DV is invariant under its
own manipulation. None of the three arms is disposition-(b) vacuous.

Two consequences of that group were designed around rather than discovered:
  * A broadcast additive constant WOULD cancel in these DVs. That is why C3
    compares RELATIVE changes of the two channels rather than raw magnitudes
    (a magnitude readout survives a broadcast constant and would give a false
    positive -- the V3-EXQ-604c shape).
  * A pooled-over-ticks correlation is a set-aggregate and IS invariant under a
    PERMUTATION of ticks. C4 (lag-1 autocorrelation) is included precisely as
    the order-SENSITIVE companion statistic, so the criterion set is not
    uniformly exchangeable.

Readiness preconditions certify these channels and no others: the harm-forward
r2 precondition certifies axis 3 ONLY; the vh-range precondition certifies the
VALENCE_HARM_DISCRIMINATIVE series ONLY; the temperature-range precondition
certifies axis 1 ONLY. None speaks for the others.

THE ACCUMULATION TRAP, AND WHY THE CRITERIA ROUTE ON INCREMENTS
--------------------------------------------------------------
ResidueField.update_valence "does NOT replace the existing value -- adds to it
so the vector accumulates across visits" (residue/field.py:294), and every
write path used here contributes a non-negative value. So the residue-field
reads are monotone non-decreasing RAMPS, not instantaneous signals: this
experiment's own Step 2.5a probe measured VALENCE_HARM_DISCRIMINATIVE going
0 -> 460.4 over 592 fresh ticks. The dACC channel is the opposite -- an
instantaneous forward-model residual that fluctuates.

Correlating a ramp against a fluctuating residual would make C1 pass (a ramp
and noise are nearly uncorrelated) and C4 pass (a ramp's lag-1 autocorrelation
is ~1, a residual's is low) for reasons that have NOTHING to do with channel
separation. That is a vacuous pass, and it would be invisible in the manifest:
the numbers would look like a clean confirmation of MECH-055.

So every criterion routes on INSTANTANEOUS quantities: the per-tick INCREMENT
of the two accumulating channels (the quantity their level integrates),
alongside the already-instantaneous temperature and harm-PE, all aligned to the
same ticks. The accumulating LEVELS are still recorded --
mean/final_valence_harm_level, and deliberately
abs_spearman_valence_harm_LEVEL_vs_dacc_pe_unrouted, the same statistic C1 uses
computed the WRONG way -- so a reader can SEE the size of the artifact the
differencing removes instead of taking it on trust. Nothing routes on them.

WHY AXIS 1's ARITHMETIC IDENTITY IS NOT A PROBLEM HERE
-----------------------------------------------------
V3-EXQ-799 warns that with mu injected into the softmax temperature, "entropy
falls as mu rises" is an arithmetic identity of the coupling. This run never
asserts that relation. It uses the axis-1 VALUE (effective_temperature, driven
upstream by PCCAnalog; pcc_stability is never written directly) purely as one
of three series in the lockstep test C2. Nothing arithmetically links the
softmax temperature to the residue-field valence store or to the harm-forward
residual, so C2 is a measurement rather than an identity. The PCC weights are
re-scoped exactly as 799 did (fatigue 0.15 / offline 0.35 / window 200)
because 799 measured that the substrate DEFAULTS pin mu at ~0.017 against a
[0,1] clip -- a structurally unsatisfiable readout. The thresholds below are
NOT relaxed to compensate.

SLEEP DRIVER: none (no sleep flags set; use_sleep_loop / sws / rem all off)

Red-team (Step 4.5, model: fable, one pass): CONTESTED -- 7 findings, 6 fixed
in this driver and 1 already addressed before the pass returned. Per-finding
dispositions are in the manifest's red_team_note; the three that changed what
this run can conclude were the verdict-grid misroute (a readiness failure was
being recorded as "FALSIFIER (ii) FIRED" / weakens), a non-causal action
pairing that trained the harm forward model on a different map from the one
the measured PE uses, and an episode-first-tick level spike in the PE series.

Claim: MECH-055 (affect.channel_separation)
Backlog: EVB-1413 (experimental twin; proposal_id EXP-0802 at authoring time,
         positional and NOT to be trusted across a governance regen)
"""

import argparse
import json
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.residue.field import (
    VALENCE_DIM,
    VALENCE_HARM_DISCRIMINATIVE,
)
from ree_core.utils.config import REEConfig

from experiment_protocol import emit_outcome
from experiments.pack_writer import write_flat_manifest, flat_readout
from experiments._lib.arm_fingerprint import arm_cell
from experiments._lib.fresh_select import FreshSelectCounter, FreshSelectProbe
from experiments._lib.precondition_gate import (
    PreconditionSpec,
    aggregate_arm_gates,
    assert_no_structurally_unsatisfiable_gate,
    evaluate_arm_gate,
)
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator

EXPERIMENT_TYPE = "v3_exq_1062_mech055_affect_channel_separation_diagnostic"
EXPERIMENT_PURPOSE = "diagnostic"
CLAIM_IDS = ["MECH-055"]
QUEUE_ID = "V3-EXQ-1062"
BACKLOG_ID = "EVB-1413"

FRESH_SELECT_NAMESPACE = "exq1062_mech055"

# --------------------------------------------------------------------------- #
# Pre-registered acceptance thresholds. Constants -- never derived from the run.
# --------------------------------------------------------------------------- #
# C1: |Spearman(VALENCE_HARM_DISCRIMINATIVE, dacc_pe)| ceiling. 0.90 is NOT
# invented here: it is SD-014's own pre-registered channel-redundancy bar
# (RHO_MAX in V3-EXQ-887/887a/887b, where |Spearman(wanting, liking)| had to
# clear <= 0.90). Same substrate, same residue-field store, same question
# shape -- "are these two channels numerically redundant".
RHO_MAX = 0.90
# C2: ceiling on the OLS R^2 between any two standardized axis series. R^2 near
# 1 is what "move in a fixed, predictable ratio" means operationally.
R2_MAX = 0.80
# C3: the PE channel's relative rise across the shift ladder must exceed the
# level channel's by this margin (both as relative change, so scale-free).
REL_CHANGE_DELTA_MIN = 0.20
# C4: the two harm representations' lag-1 autocorrelations must differ by this
# much. An RBF-smoothed EMA level and a forward-model residual have different
# temporal signatures; one collapsed scalar wearing two labels cannot.
LAG1_DELTA_MIN = 0.10
# Per-arm seed quorum for every criterion.
SEED_QUORUM = 2

# --------------------------------------------------------------------------- #
# Readiness floors (substrate-not-ready, NOT claim verdicts).
# --------------------------------------------------------------------------- #
# Axis 3 must be a genuine forward-model PE. Below this the dACC "PE" degenerates
# to ||z_harm_a|| (dacc._affective_pe's own z_harm_a_pred-is-None branch) and
# axis 3 does not exist as a distinct channel at all.
FORWARD_R2_MIN = 0.30
# Each series must actually vary, or its correlation/R^2 is undefined rather
# than low. These are floors on the statistic the criteria route on (RANGE of
# the series), not on a magnitude proxy for it.
VH_RANGE_MIN = 1e-4
TEMP_RANGE_MIN = 1e-6
PE_RANGE_MIN = 1e-4
# The level channel is only a matched control if harm exposure really is
# matched across arms. CEILING on each arm's relative deviation from the
# pooled mean.
HARM_EXPOSURE_REL_DEV_MAX = 0.25
# The IV must actually have moved the PE in the shift arms (scoped OUT of the
# stationary arm, which IS the reference).
PE_ELEVATION_MIN = 0.05
# Sample floor for the per-tick statistics, counted in FRESH E3 selections.
FRESH_TICKS_MIN = 200
CORR_MIN_N = 100

# --------------------------------------------------------------------------- #
# Schedule. Phased training is MANDATORY: E2HarmAForward trains on z_harm_a,
# an encoder output, so P0 warms the encoder with no downstream loss, P1
# freezes it and trains the forward model on .detach()ed latents, P2 measures
# with no training at all.
# --------------------------------------------------------------------------- #
# Sized from the Step 2.5a measured cost (~0.84 s CPU per env step on this
# config) against the fleet's observed ~0.4 s/step for comparable heavy runs
# (V3-EXQ-1050: 6 cells, 330 estimated minutes). P1 gets twice P0's budget
# because P1 is what the harm-forward model -- the axis-3 readiness gate, and
# therefore the existence of axis 3 at all -- actually needs.
P0_EPS = 30
P1_EPS = 60
TOTAL_TRAINING_EPS = P0_EPS + P1_EPS      # the [train] ep N/M denominator
STEPS_PER_EPISODE = 90
# P2 is a FIXED STEP BUDGET, not an episode count. Episodes end early in the
# harsher shift arms, so an episode-count loop would give the arms different
# step counts and the per-tick series would differ in LENGTH mechanically --
# manufacturing a difference with no bearing on channel separation. (The same
# reasoning V3-EXQ-799 applied to visitation entropy.)
# 1800 steps at the MEASURED fresh-select yield (592/600 = 98.7% once the
# valence write paths are driven) projects to ~1780 fresh selections per cell,
# far above FRESH_TICKS_MIN and CORR_MIN_N.
P2_STEP_BUDGET = 1800
EPSILON_TRAIN = 0.1
EPSILON_EVAL = 0.0

SEEDS = [42, 137, 2026]

ARMS: List[Dict[str, Any]] = [
    {"arm_id": "ARM_0_STATIONARY", "interval": 0, "is_shift": False},
    {"arm_id": "ARM_2_HIGH_SHIFT", "interval": 10, "is_shift": True},
]
STATIONARY_ARM = "ARM_0_STATIONARY"
HIGH_ARM = "ARM_2_HIGH_SHIFT"
WORLD_RULE_SHIFT_DEPTH = 2

# Substrate dims / indices.
WORLD_DIM = 32
SELF_DIM = 32
HARM_A_DIM = 16
IDX_HARM_EXPOSURE = 10
IDX_BENEFIT_EXPOSURE = 11

# Env config. 887b's validated valence-population env (hazard_harm 0.5,
# resource_benefit 0.3, num_resources 6) -- deliberately NOT a weak-signal
# config, which would empty the LIKING/HARM channels outright (887b's own note
# on the V3-EXQ-432 vacuous-zero failure mode). harm_history_len=10 gives the
# affective stream its temporal integration input. env_drift PINNED OFF.
ENV_KWARGS = dict(
    size=10,
    num_hazards=3,
    num_resources=6,
    hazard_harm=0.5,
    resource_benefit=0.3,
    use_proxy_fields=True,
    harm_history_len=10,
    env_drift_interval=999,
    env_drift_prob=0.0,
)

# Substrate gate: what counts as consummatory contact. Calibrated to the
# measured benefit_exposure distribution on this env (887b measured max ~0.059,
# so the substrate default 0.1 is structurally unreachable). NOT an acceptance
# threshold.
LIKING_THRESHOLD = 0.02

# SD-014 incentive-sensitization at 887b's PASSING gain. 887b (2026-08-08)
# PASSED with rate 0.10 / max 8.0 / coupling 2.5 where 887a FAILED at the
# substrate defaults; that is what makes the valence store non-degenerate
# rather than wanting/liking-collinear.
SENSITIZATION_RATE = 0.10
SENSITIZATION_MAX = 8.0
SENSITIZATION_COUPLING = 2.5

# dACC. The bias multipliers must be non-zero or DACCtoE3Adapter returns the
# zero vector regardless of bundle content (its own docstring) -- and
# dacc_foraging_weight is DELIBERATELY 0.0 because foraging_value is a
# BROADCAST SCALAR, the exact V3-EXQ-604c DV-symmetry vacuity shape. Values
# follow 799.
DACC_WEIGHT = 1.0
DACC_INTERACTION_WEIGHT = 0.5
DACC_BIAS_MAX_ABS = 1.0
DACC_PRECISION_SCALE = 5000.0

# PCC re-scoping, verbatim from 799: the substrate defaults pin mu at ~0.017.
PCC_FATIGUE_WEIGHT = 0.15
PCC_OFFLINE_WEIGHT = 0.35
PCC_OFFLINE_RECENCY_WINDOW = 200

E2_HARM_A_LR = 5e-4

AXIS_NAMES = ["axis1_temperature", "axis2_valence_spread", "axis3_harm_pe"]

_ZG = ZGoalStreamAccumulator()


# --------------------------------------------------------------------------- #
# statistics                                                                  #
# --------------------------------------------------------------------------- #

def _rank(xs: List[float]) -> List[float]:
    """Average-tie ranks."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    if n < CORR_MIN_N or n != len(ys):
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx < 1e-12 or dy < 1e-12:
        return float("nan")
    return num / (dx * dy)


def _spearman(xs: List[float], ys: List[float]) -> float:
    if len(xs) < CORR_MIN_N or len(xs) != len(ys):
        return float("nan")
    return _pearson(_rank(xs), _rank(ys))


def _r2(xs: List[float], ys: List[float]) -> float:
    """OLS R^2 between two series. Equal to pearson^2 for a simple fit."""
    r = _pearson(xs, ys)
    return float("nan") if math.isnan(r) else r * r


def _lag1_autocorr(xs: List[float]) -> float:
    if len(xs) < CORR_MIN_N + 1:
        return float("nan")
    return _pearson(xs[:-1], xs[1:])


def _series_range(xs: List[float]) -> float:
    finite = [x for x in xs if math.isfinite(x)]
    if not finite:
        return 0.0
    return float(max(finite) - min(finite))


def _mean(xs: List[float]) -> float:
    finite = [x for x in xs if math.isfinite(x)]
    return float(sum(finite) / len(finite)) if finite else float("nan")


def _rel_change(new: float, ref: float) -> float:
    """Relative change, scale-free. Guarded denominator."""
    if not (math.isfinite(new) and math.isfinite(ref)):
        return float("nan")
    return (new - ref) / (abs(ref) + 1e-9)


def _worst_cell(rows: List[Dict], key: str, mode: str) -> Tuple[float, str]:
    """Return (extremum, offending_cell_id). mode 'min' for floors, 'max' for ceilings."""
    vals = [(r[key], r["cell_id"]) for r in rows if math.isfinite(r.get(key, float("nan")))]
    if not vals:
        return float("nan"), "(no finite cell)"
    pick = min(vals, key=lambda t: t[0]) if mode == "min" else max(vals, key=lambda t: t[0])
    return float(pick[0]), str(pick[1])


# --------------------------------------------------------------------------- #
# setup                                                                       #
# --------------------------------------------------------------------------- #

def _make_env(seed: int, interval: int) -> CausalGridWorldV2:
    kw = dict(ENV_KWARGS)
    kw.update(
        world_rule_shift_enabled=(interval > 0),
        world_rule_shift_interval=interval,
        world_rule_shift_depth=WORLD_RULE_SHIFT_DEPTH if interval > 0 else 0,
    )
    return CausalGridWorldV2(seed=seed, **kw)


def make_config(env: CausalGridWorldV2) -> REEConfig:
    """Identical in every arm. The ONLY cross-arm difference is the ENV's
    world_rule_shift rate -- no agent-side flag varies, so an arm difference
    cannot be an agent-config artifact."""
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        alpha_world=0.9,
        # SD-011 dual nociceptive streams: z_harm_s and z_harm_a, the two
        # upstream sources whose downstream readouts C1 compares.
        use_harm_stream=True,
        harm_obs_dim=51,
        use_affective_harm_stream=True,
        harm_obs_a_dim=50,
        z_harm_a_dim=HARM_A_DIM,
        harm_history_len=10,
        # Axis 3: MECH-258 precision-weighted affective-pain PE.
        use_e2_harm_a=True,
        use_shared_harm_trunk=False,
        e2_harm_a_lr=E2_HARM_A_LR,
        use_dacc=True,
        dacc_weight=DACC_WEIGHT,
        dacc_interaction_weight=DACC_INTERACTION_WEIGHT,
        dacc_foraging_weight=0.0,
        dacc_bias_max_abs=DACC_BIAS_MAX_ABS,
        dacc_precision_scale=DACC_PRECISION_SCALE,
        dacc_effort_cost=0.1,
        dacc_drive_coupling=0.0,
        # Axis 1: SD-032d mu/kappa -> mode-prior softmax temperature.
        use_salience_coordinator=True,
        use_pcc_analog=True,
        salience_apply_to_dacc_bias=True,
        salience_use_stability_temperature=True,
        salience_temperature_mu_alpha=1.0,
        salience_temperature_kappa_alpha=0.0,
        salience_temperature_exponent_clip=4.0,
        pcc_fatigue_weight=PCC_FATIGUE_WEIGHT,
        pcc_offline_weight=PCC_OFFLINE_WEIGHT,
        pcc_offline_recency_window=PCC_OFFLINE_RECENCY_WINDOW,
        # Axis 2: SD-014 valence vector, at 887b's passing sensitization gain.
        z_goal_enabled=True,
        drive_weight=2.0,
        benefit_eval_enabled=True,
        goal_weight=1.0,
        tonic_5ht_enabled=True,
        valence_harm_enabled=True,
        valence_liking_enabled=True,
        liking_threshold=LIKING_THRESHOLD,
        surprise_gated_replay=True,
        pe_surprise_threshold=0.001,
        incentive_sensitization_enabled=True,
        sensitization_rate=SENSITIZATION_RATE,
        sensitization_max=SENSITIZATION_MAX,
        sensitization_coupling=SENSITIZATION_COUPLING,
    )
    cfg.latent.use_resource_encoder = True
    return cfg


def _obs_tensors(obs_dict) -> Tuple[torch.Tensor, ...]:
    body = obs_dict["body_state"].float().unsqueeze(0)
    world = obs_dict["world_state"].float().unsqueeze(0)
    harm = obs_dict["harm_obs"].float().unsqueeze(0) if "harm_obs" in obs_dict else None
    harm_a = obs_dict["harm_obs_a"].float().unsqueeze(0) if "harm_obs_a" in obs_dict else None
    hh = obs_dict["harm_history"].float().unsqueeze(0) if "harm_history" in obs_dict else None
    return body, world, harm, harm_a, hh


def _drive_valence_write_paths(agent, body: torch.Tensor) -> Tuple[float, float]:
    """Drive the SD-014 valence write paths (887b's sequence). Returns
    (benefit_exposure, harm_exposure). Without these the residue field never
    allocates an active RBF center and evaluate_valence returns all zeros --
    confirmed by this experiment's own Step 2.5a probe."""
    be = float(body[0, IDX_BENEFIT_EXPOSURE])
    he = float(body[0, IDX_HARM_EXPOSURE])
    drive = agent.compute_drive_level(body)
    agent.serotonin_step(be)
    agent.update_z_goal(be, drive)
    agent.update_benefit_salience(be, drive)   # -> VALENCE_WANTING
    agent.update_harm_salience(he)             # -> VALENCE_HARM_DISCRIMINATIVE
    agent.update_liking(be)                    # -> VALENCE_LIKING
    return be, he


def _cross_candidate_valence_range(agent, candidates, index: int) -> float:
    """Max over VALENCE components of the cross-candidate range of
    evaluate_valence at the candidates' world_states[index].

    index=-1 is the POST-ACTION terminus; index=0 is the rollout's SHARED
    initial z_world seed, which E2FastPredictor.rollout_with_world makes
    bit-identical across every candidate by construction (config.py's
    candidate_summary_source note; measured 2.8e6-4.5e6 magnitude ratio in
    V3-EXQ-822c). So index=0 is recorded as a NEGATIVE CONTROL that must read
    ~0, and index=-1 is the informative read. Returns nan when unavailable."""
    zs = []
    for c in candidates:
        ws = getattr(c, "world_states", None)
        if ws:
            zs.append(ws[index].detach().reshape(1, -1))
    if len(zs) < 2:
        return float("nan")
    batch = torch.cat(zs, dim=0)
    vv = agent.residue_field.evaluate_valence(batch)
    if vv.dim() != 2 or vv.shape[0] != batch.shape[0]:
        return float("nan")
    return float((vv.max(dim=0).values - vv.min(dim=0).values).max().item())


# --------------------------------------------------------------------------- #
# one (arm x seed) cell                                                       #
# --------------------------------------------------------------------------- #

def _config_slice(interval: int) -> Dict[str, Any]:
    """What the cell's computation reads. Declared for the arm fingerprint."""
    return {
        "env": dict(ENV_KWARGS),
        "world_rule_shift_interval": interval,
        "world_rule_shift_depth": WORLD_RULE_SHIFT_DEPTH if interval > 0 else 0,
        "schedule": {
            "p0": P0_EPS, "p1": P1_EPS, "steps": STEPS_PER_EPISODE,
            "p2_step_budget": P2_STEP_BUDGET,
            "epsilon_train": EPSILON_TRAIN, "epsilon_eval": EPSILON_EVAL,
        },
        "substrate": {
            "world_dim": WORLD_DIM, "self_dim": SELF_DIM, "z_harm_a_dim": HARM_A_DIM,
            "dacc_weight": DACC_WEIGHT,
            "dacc_interaction_weight": DACC_INTERACTION_WEIGHT,
            "dacc_bias_max_abs": DACC_BIAS_MAX_ABS,
            "dacc_precision_scale": DACC_PRECISION_SCALE,
            "pcc_fatigue_weight": PCC_FATIGUE_WEIGHT,
            "pcc_offline_weight": PCC_OFFLINE_WEIGHT,
            "pcc_offline_recency_window": PCC_OFFLINE_RECENCY_WINDOW,
            "sensitization_rate": SENSITIZATION_RATE,
            "sensitization_max": SENSITIZATION_MAX,
            "sensitization_coupling": SENSITIZATION_COUPLING,
            "liking_threshold": LIKING_THRESHOLD,
            "e2_harm_a_lr": E2_HARM_A_LR,
        },
        # Readout-affecting constants, declared so a cross-driver consumer with
        # different values MISSES this mint rather than falsely HITting it
        # (arm_reuse_fingerprint_plan.md 7b; confirmed instance V3-EXQ-798).
        # CORR_MIN_N gates every correlation estimator in this cell -- a
        # different value changes which cells return nan; the IDX_* constants
        # select which body_obs slots drive the valence write paths and the
        # harm-exposure series, so a different indexing scheme silently
        # measures different quantities under the same fingerprint.
        "readout_constants": {
            "corr_min_n": CORR_MIN_N,
            "idx_harm_exposure": IDX_HARM_EXPOSURE,
            "idx_benefit_exposure": IDX_BENEFIT_EXPOSURE,
        },
    }


def _run_cell(arm: Dict[str, Any], seed: int, dry_run: bool) -> Dict[str, Any]:
    arm_id = arm["arm_id"]
    interval = int(arm["interval"])
    cell_id = f"{arm_id}/seed{seed}"
    print("Seed %d Condition %s" % (seed, arm_id), flush=True)

    p0_eps = 2 if dry_run else P0_EPS
    p1_eps = 2 if dry_run else P1_EPS
    steps_per_ep = 12 if dry_run else STEPS_PER_EPISODE
    p2_budget = 60 if dry_run else P2_STEP_BUDGET
    total_training = p0_eps + p1_eps

    with arm_cell(
        seed,
        config_slice=_config_slice(interval),
        script_path=Path(__file__),
        config_slice_declared=True,
        include_driver_script_in_hash=False,
    ) as cell:
        random.seed(seed)
        env = _make_env(seed, interval)
        agent = REEAgent(make_config(env))

        e2a_opt = optim.Adam(agent.e2_harm_a.parameters(), lr=E2_HARM_A_LR)
        e1_opt = optim.Adam(list(agent.e1.parameters()), lr=1e-3)
        wf_opt = optim.Adam(
            list(agent.e2.world_transition.parameters())
            + list(agent.e2.world_action_encoder.parameters()),
            lr=1e-3,
        )

        probe = FreshSelectProbe(FRESH_SELECT_NAMESPACE)
        counter = FreshSelectCounter()

        wf_buf: List[Tuple] = []
        max_buf = 2000

        # --------------------------- P0 + P1 ------------------------------- #
        for ep in range(total_training):
            is_p0 = ep < p0_eps
            agent.reset()
            _obs, od = env.reset()
            prev_zha: Optional[torch.Tensor] = None
            prev_zw: Optional[torch.Tensor] = None
            prev_action: Optional[torch.Tensor] = None

            if (ep + 1) % 20 == 0 or ep == 0:
                print(
                    "  [train] %s seed=%d ep %d/%d" % (arm_id, seed, ep + 1, total_training),
                    flush=True,
                )

            for _step in range(steps_per_ep):
                body, world, harm, harm_a, hh = _obs_tensors(od)
                latent = agent.sense(
                    obs_body=body, obs_world=world, obs_harm=harm,
                    obs_harm_a=harm_a, obs_harm_history=hh,
                )
                ticks = agent.clock.advance()
                e1_prior = (
                    agent._e1_tick(latent) if ticks.get("e1_tick")
                    else torch.zeros(1, WORLD_DIM, device=agent.device)
                )
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                action = agent.select_action(candidates, ticks)

                if EPSILON_TRAIN > 0.0 and random.random() < EPSILON_TRAIN:
                    ai = random.randint(0, env.action_dim - 1)
                    action = torch.zeros(1, env.action_dim, device=agent.device)
                    action[0, ai] = 1.0

                zw_curr = latent.z_world.detach()

                # P0: encoder warmup only -- E1 + world-forward. No downstream
                # head sees a gradient here (phased training, MANDATORY).
                if is_p0:
                    e1_loss = agent.compute_prediction_loss()
                    if e1_loss.requires_grad:
                        e1_opt.zero_grad()
                        e1_loss.backward()
                        torch.nn.utils.clip_grad_norm_(agent.e1.parameters(), 1.0)
                        e1_opt.step()
                    if prev_zw is not None and prev_action is not None:
                        wf_buf.append((prev_zw.cpu(), prev_action.cpu(), zw_curr.cpu()))
                        if len(wf_buf) > max_buf:
                            wf_buf = wf_buf[-max_buf:]
                    if len(wf_buf) >= 16:
                        k = min(32, len(wf_buf))
                        idx = torch.randperm(len(wf_buf))[:k].tolist()
                        zb = torch.cat([wf_buf[i][0] for i in idx]).to(agent.device)
                        ab = torch.cat([wf_buf[i][1] for i in idx]).to(agent.device)
                        zn = torch.cat([wf_buf[i][2] for i in idx]).to(agent.device)
                        wf_l = F.mse_loss(agent.e2.world_forward(zb, ab), zn)
                        if wf_l.requires_grad:
                            wf_opt.zero_grad()
                            wf_l.backward()
                            torch.nn.utils.clip_grad_norm_(
                                list(agent.e2.world_transition.parameters())
                                + list(agent.e2.world_action_encoder.parameters()), 1.0)
                            wf_opt.step()

                # P1: encoder FROZEN by .detach() on both sides; only the harm
                # forward model learns.
                #
                # THE ACTION IS prev_action, NOT action. The transition
                # z_harm_a(t-1) -> z_harm_a(t) was caused by the action executed
                # at t-1; pairing it with THIS tick's action trains a
                # non-causal map. It also has to match the pairing the agent's
                # own measured PE uses, or P1 trains a different model from the
                # one C1 reads: sense() caches _harm_a_prev = z(t)
                # (agent.py:5583), select_action rolls
                # pred = e2_harm_a(z(t), a(t)) (agent.py:10329), and the next
                # tick's dACC compares z(t+1) against it (agent.py:7743).
                # Since world_rule_shift acts ONLY through the action channel
                # (causal_grid_world.py re-permutes the action -> displacement
                # map), training against the wrong action would blunt exactly
                # the manipulation this experiment depends on.
                if ((not is_p0) and prev_zha is not None
                        and prev_action is not None and latent.z_harm_a is not None):
                    z_pred = agent.e2_harm_a(prev_zha.detach(), prev_action.detach())
                    loss = agent.e2_harm_a.compute_loss(z_pred, latent.z_harm_a.detach())
                    if loss.requires_grad:
                        e2a_opt.zero_grad()
                        loss.backward()
                        e2a_opt.step()

                _drive_valence_write_paths(agent, body)
                _obs, harm_signal, done, _info, od = env.step(
                    int(action.argmax(dim=-1).item()))
                agent.update_residue(float(harm_signal) if harm_signal is not None else 0.0)

                prev_zha = (latent.z_harm_a.detach().clone()
                            if latent.z_harm_a is not None else None)
                prev_zw = zw_curr
                prev_action = action.detach()
                if done:
                    break

        # ------------------------------ P2 --------------------------------- #
        # Measurement only. No optimiser is stepped. Fixed step budget.
        #
        # Series are collected PER EPISODE and differenced WITHIN an episode,
        # for two reasons found by the Step 4.5 red-team pass and confirmed in
        # ree_core:
        #   (a) agent.reset() clears _harm_a_pred_prev (agent.py:3630), so on
        #       the FIRST fresh tick of every episode the dACC receives
        #       z_harm_a_pred=None and dacc._affective_pe returns ||z_harm_a||
        #       -- a LEVEL, not a residual (dacc.py:213-214). Those ticks are
        #       EXCLUDED and counted, because a level is ~an order of magnitude
        #       above a residual and episode counts differ across arms (shift
        #       arms end episodes earlier), so including them would let
        #       "PE load elevated" and C3 pass on episode count alone.
        #   (b) differencing across an episode boundary would pair an increment
        #       spanning a reset with a post-reset instantaneous value.
        ep_series: List[Dict[str, List[float]]] = []
        hs_norm: List[float] = []
        ha_norm: List[float] = []
        prec: List[float] = []
        prec_norm_vals: List[float] = []
        # Recorded, NON-GATING cross-candidate valence range. Flat (not
        # per-episode / not differenced) because the consumers are plain means
        # of a LEVEL-family read -- the same treatment prec / hs_norm / ha_norm
        # get, not the _diff() treatment vh / a2 get.
        cc_post: List[float] = []
        cc_seed: List[float] = []
        harm_exposure: List[float] = []
        r2_pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
        total_steps = 0
        n_episodes = 0
        n_level_pe_ticks_excluded = 0

        with torch.no_grad():
            while total_steps < p2_budget:
                agent.reset()
                _obs, od = env.reset()
                counter.flush()
                n_episodes += 1
                prev_zha = None
                prev_action = None
                cur: Dict[str, List[float]] = {"vh": [], "a2": [], "pe": [], "a1": []}
                while total_steps < p2_budget:
                    body, world, harm, harm_a, hh = _obs_tensors(od)
                    latent = agent.sense(
                        obs_body=body, obs_world=world, obs_harm=harm,
                        obs_harm_a=harm_a, obs_harm_history=hh,
                    )
                    ticks = agent.clock.advance()
                    e1_prior = (
                        agent._e1_tick(latent) if ticks.get("e1_tick")
                        else torch.zeros(1, WORLD_DIM, device=agent.device)
                    )
                    candidates = agent.generate_trajectories(latent, e1_prior, ticks)

                    # Read BEFORE select_action: this is what the dACC inside
                    # that call will receive as z_harm_a_pred. None => the pe it
                    # reports is a level, not a residual (see (a) above).
                    pred_absent = getattr(agent, "_harm_a_pred_prev", None) is None

                    # Sample-size integrity: E3 diagnostics LATCH between e3
                    # ticks (cadence heartbeat.e3_steps_per_tick, default 10),
                    # so a per-env-step read without this guard would
                    # pseudo-replicate ~10x. n_latched_ticks is emitted.
                    with probe.watch(agent) as fresh:
                        action = agent.select_action(candidates, ticks)
                    is_fresh = bool(fresh)
                    counter.record(is_fresh)

                    if is_fresh:
                        b = getattr(agent, "_dacc_last_bundle", None)
                        st = getattr(agent, "_salience_last_tick", None)
                        if isinstance(b, dict) and "pe" in b and isinstance(st, dict):
                            if pred_absent:
                                n_level_pe_ticks_excluded += 1
                            else:
                                v = agent.residue_field.evaluate_valence(
                                    latent.z_world).reshape(-1)
                                cur["pe"].append(float(b["pe"]))
                                cur["a1"].append(
                                    float(st.get("effective_temperature", float("nan"))))
                                # axis 2 scalar: the SPREAD across valence
                                # components at the realized z_world -- the
                                # not-collapsed-to-a-scalar quantity itself.
                                cur["a2"].append(float((v.max() - v.min()).item()))
                                cur["vh"].append(
                                    float(v[VALENCE_HARM_DISCRIMINATIVE].item()))
                                # Recorded, NON-GATING. Sampled on exactly the
                                # ticks every other per-tick DV in this row is
                                # sampled on -- FRESH E3 selections with a real
                                # (non-level) PE -- so the recorded mean
                                # describes the same tick population the routed
                                # statistics do. The fresh gate is the
                                # pseudo-replication guard documented above; the
                                # recorded_non_gating_note's own Step 2.5a
                                # figure is stated "over 592 fresh ticks".
                                # index=-1 post-action terminus, index=0 shared
                                # seed -- see _cross_candidate_valence_range.
                                cc_post.append(
                                    _cross_candidate_valence_range(
                                        agent, candidates, -1))
                                cc_seed.append(
                                    _cross_candidate_valence_range(
                                        agent, candidates, 0))
                                _pv = float(getattr(agent.e3, "current_precision",
                                                    float("nan")))
                                prec.append(_pv)
                                # dacc._affective_pe applies
                                # prec_norm = min(precision / dacc_precision_scale, 3.0)
                                # and multiplies the residual by (1 + prec_norm).
                                # Recorded, not gated: every routed statistic here
                                # (Spearman, R^2, lag-1 autocorr, relative change) is
                                # invariant under a positive constant scaling, so
                                # neither a negligible nor a saturated precision leg
                                # can manufacture or destroy a verdict -- but both
                                # bound what a PASS may be said to have exercised.
                                if math.isfinite(_pv):
                                    prec_norm_vals.append(
                                        min(_pv / DACC_PRECISION_SCALE, 3.0))
                                if latent.z_harm is not None:
                                    hs_norm.append(float(latent.z_harm.norm().item()))
                                if latent.z_harm_a is not None:
                                    ha_norm.append(float(latent.z_harm_a.norm().item()))
                        # r2 on the SAME causal pairing the agent's own PE uses:
                        # sense() caches _harm_a_prev = z(t) (agent.py:5583) and
                        # select_action rolls e2_harm_a(z(t), a(t)) -> z(t+1)
                        # (agent.py:10329). So the predecessor ACTION is
                        # prev_action, never this tick's action.
                        if (prev_zha is not None and prev_action is not None
                                and latent.z_harm_a is not None):
                            z_pred = agent.e2_harm_a(prev_zha.detach(),
                                                     prev_action.detach())
                            r2_pairs.append((z_pred.detach().cpu(),
                                             latent.z_harm_a.detach().cpu()))

                    harm_exposure.append(float(body[0, IDX_HARM_EXPOSURE]))
                    _drive_valence_write_paths(agent, body)
                    _obs, harm_signal, done, _info, od = env.step(
                        int(action.argmax(dim=-1).item()))
                    agent.update_residue(
                        float(harm_signal) if harm_signal is not None else 0.0)
                    prev_zha = (latent.z_harm_a.detach().clone()
                                if latent.z_harm_a is not None else None)
                    prev_action = action.detach()
                    total_steps += 1
                    if done:
                        break
                ep_series.append(cur)
        counter.flush()
        _ZG.observe(agent)

        # Per-tick increments of the accumulating channels, differenced WITHIN
        # each episode and then concatenated. These -- not the levels -- are
        # what C1 / C2 / C3 / C4 route on (see the P2 header note).
        def _diff(xs: List[float]) -> List[float]:
            return [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]

        vh: List[float] = []
        a2: List[float] = []
        pe_al: List[float] = []
        a1_al: List[float] = []
        vh_level: List[float] = []
        a2_level: List[float] = []
        pe: List[float] = []
        for ep in ep_series:
            vh_level.extend(ep["vh"])
            a2_level.extend(ep["a2"])
            pe.extend(ep["pe"])
            if len(ep["vh"]) < 2:
                continue
            vh.extend(_diff(ep["vh"]))
            a2.extend(_diff(ep["a2"]))
            # Align the instantaneous series to the differenced ones by
            # dropping each episode's first sample, so every pairwise
            # statistic below is computed on the SAME ticks.
            pe_al.extend(ep["pe"][1:])
            a1_al.extend(ep["a1"][1:])


        # harm-forward r2, on the P2 pairs (597b's estimator).
        if r2_pairs:
            preds = torch.cat([p for p, _ in r2_pairs])
            tgts = torch.cat([t for _, t in r2_pairs])
            ss_res = float(((tgts - preds) ** 2).sum())
            ss_tot = float(((tgts - tgts.mean(dim=0)) ** 2).sum())
            fwd_r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-8 else float("nan")
        else:
            fwd_r2 = float("nan")

        row: Dict[str, Any] = {
            "cell_id": cell_id,
            "arm_id": arm_id,
            "seed": seed,
            "world_rule_shift_interval": interval,
            "is_shift_arm": bool(arm["is_shift"]),
            "n_fresh_select": counter.n_fresh_select,
            "n_latched_ticks": counter.n_latched,
            "n_p2_env_steps": total_steps,
            "n_p2_episodes": n_episodes,
            "n_level_pe_ticks_excluded": n_level_pe_ticks_excluded,
            "harm_a_forward_r2": fwd_r2,
            # series summaries
            "mean_dacc_pe": _mean(pe),
            # The level channel's INTENSITY under a fixed step budget is its
            # accumulation RATE, i.e. the mean per-tick increment. C3 routes on
            # this, not on the level (a level mean over a ramp is a function of
            # where the window happens to sit).
            "mean_valence_harm_delta": _mean(vh),
            "mean_axis1_temperature": _mean(a1_al),
            "mean_axis2_valence_spread_delta": _mean(a2),
            # recorded, NOT routed -- accumulating levels (see note above)
            "mean_valence_harm_level": _mean(vh_level),
            "final_valence_harm_level": vh_level[-1] if vh_level else float("nan"),
            "mean_axis2_valence_spread_level": _mean(a2_level),
            "mean_harm_exposure": _mean(harm_exposure),
            "mean_z_harm_s_norm": _mean(hs_norm),
            "mean_z_harm_a_norm": _mean(ha_norm),
            "mean_e3_precision": _mean(prec),
            # The ACTUAL precision contribution, not just its saturation: the
            # dACC multiplies the residual by (1 + prec_norm). Near 0 the
            # precision leg is negligible; at the 3.0 cap it is a constant.
            # BOTH extremes make it inert, and only the middle exercises it.
            "mean_prec_norm": _mean(prec_norm_vals),
            "frac_precision_weight_saturated": _mean(
                [1.0 if x >= 3.0 else 0.0 for x in prec_norm_vals]),
            "range_dacc_pe": _series_range(pe),
            "range_valence_harm_delta": _series_range(vh),
            "range_axis1_temperature": _series_range(a1_al),
            "range_axis2_valence_spread_delta": _series_range(a2),
            "range_valence_harm_level": _series_range(vh_level),
            # C1
            "abs_spearman_valence_harm_vs_dacc_pe": abs(_spearman(vh, pe_al))
            if not math.isnan(_spearman(vh, pe_al)) else float("nan"),
            # Recorded for contrast ONLY, never routed: the same statistic on
            # the accumulating LEVEL. If this differs wildly from the routed
            # value, that difference IS the ramp artifact the differencing
            # removes, and a reader can see it rather than having to trust it.
            "abs_spearman_valence_harm_LEVEL_vs_dacc_pe_unrouted": abs(
                _spearman(vh_level, pe)) if not math.isnan(
                _spearman(vh_level, pe)) else float("nan"),
            # C2 -- all three pairs, all on instantaneous quantities
            "r2_axis1_axis2": _r2(a1_al, a2),
            "r2_axis1_axis3": _r2(a1_al, pe_al),
            "r2_axis2_axis3": _r2(a2, pe_al),
            # C4
            "lag1_dacc_pe": _lag1_autocorr(pe_al),
            "lag1_valence_harm": _lag1_autocorr(vh),
            "lag1_valence_harm_LEVEL_unrouted": _lag1_autocorr(vh_level),
            # recorded, NON-GATING
            "cross_candidate_valence_range_post_action_mean": _mean(cc_post),
            "cross_candidate_valence_range_shared_seed_mean": _mean(cc_seed),
            "n_series": len(pe_al),
        }
        pairs = [row["r2_axis1_axis2"], row["r2_axis1_axis3"], row["r2_axis2_axis3"]]
        finite_pairs = [p for p in pairs if math.isfinite(p)]
        row["max_pairwise_axis_r2"] = max(finite_pairs) if finite_pairs else float("nan")
        row["lag1_abs_diff_pe_vs_valence_harm"] = (
            abs(row["lag1_dacc_pe"] - row["lag1_valence_harm"])
            if math.isfinite(row["lag1_dacc_pe"]) and math.isfinite(row["lag1_valence_harm"])
            else float("nan")
        )
        cell.stamp(row)

    # Per-cell criterion flags (C3 is cross-arm and is decided in _run).
    row["c1_pass"] = bool(
        math.isfinite(row["abs_spearman_valence_harm_vs_dacc_pe"])
        and row["abs_spearman_valence_harm_vs_dacc_pe"] <= RHO_MAX
    )
    row["c2_pass"] = bool(
        math.isfinite(row["max_pairwise_axis_r2"])
        and row["max_pairwise_axis_r2"] <= R2_MAX
    )
    row["c4_pass"] = bool(
        math.isfinite(row["lag1_abs_diff_pe_vs_valence_harm"])
        and row["lag1_abs_diff_pe_vs_valence_harm"] >= LAG1_DELTA_MIN
    )
    cell_pass = row["c1_pass"] and row["c2_pass"]
    print("verdict: %s" % ("PASS" if cell_pass else "FAIL"), flush=True)
    return row


# --------------------------------------------------------------------------- #
# readiness preconditions                                                     #
# --------------------------------------------------------------------------- #

def _precondition_specs() -> List[PreconditionSpec]:
    return [
        PreconditionSpec(
            name="harm_a_forward_r2_supra_floor",
            description=(
                "E2HarmAForward must actually predict z_harm_a on held-out P2 "
                "transitions. Below this floor dacc._affective_pe falls back to "
                "||z_harm_a|| (its z_harm_a_pred-is-None branch), axis 3 is a LEVEL "
                "rather than a forward-model residual, and C1 would be comparing two "
                "levels -- a starved criterion, not a falsified one."),
            control=(
                "V3-EXQ-597b measured harm_a_forward_r2=0.91 on its PE_FORWARD arm on "
                "this substrate family against the same 0.30 floor; worst seed in this "
                "arm is reported."),
            threshold=FORWARD_R2_MIN,
            direction="lower",
        ),
        PreconditionSpec(
            name="valence_harm_series_range_supra_floor",
            description=(
                "Cross-tick RANGE of the PER-TICK INCREMENT of the "
                "VALENCE_HARM_DISCRIMINATIVE read -- deliberately the increment and "
                "not the level. The residue field ACCUMULATES (field.py:294: update_"
                "valence 'does NOT replace the existing value -- adds to it'), so the "
                "level is a monotone ramp whose range is large no matter what the "
                "channel is doing; the increment is the instantaneous quantity C1/C3/C4 "
                "actually route on. This is therefore the SAME statistic those criteria "
                "consume: a near-constant increment series gives an undefined "
                "correlation, which must read not-ready rather than low."),
            control=(
                "V3-EXQ-887b populated HARM_DISC on 31/32 nodes at this exact env and "
                "write-path config. This experiment's own 2026-09-19 Step 2.5a probe "
                "measured a LEVEL range of EXACTLY 0 until the update_residue / "
                "update_harm_salience write paths were driven, and 460.4 once they "
                "were -- which is why they are driven explicitly here and why this "
                "precondition gates. The floor is applied to the INCREMENT range, "
                "which the probe's 0 -> 460.4 ramp over 592 ticks implies is "
                "comfortably non-zero (mean increment ~0.78) but which is measured "
                "here rather than inferred."),
            threshold=VH_RANGE_MIN,
            direction="lower",
        ),
        PreconditionSpec(
            name="dacc_pe_series_range_supra_floor",
            description=(
                "Cross-tick RANGE of the dACC precision-weighted harm PE -- the same "
                "statistic C1/C2/C4 route on, for the same reason as above."),
            control=(
                "Step 2.5a probe, 2026-09-19: bundle['pe'] range 0.771 over 592 fresh "
                "E3 ticks on this exact config with the write paths driven."),
            threshold=PE_RANGE_MIN,
            direction="lower",
        ),
        PreconditionSpec(
            name="axis1_temperature_range_supra_floor",
            description=(
                "Cross-tick RANGE of effective_temperature (axis 1's value). C2's "
                "R^2 against axis 1 is undefined if it never varies."),
            control=(
                "Step 2.5a probe, 2026-09-19: effective_temperature range 0.310 and "
                "pcc_stability 0.0855-0.498 (range 0.413) under 799's re-scoped PCC "
                "weights -- NOT the ~0.017 pinned value 799 measured at the substrate "
                "defaults."),
            threshold=TEMP_RANGE_MIN,
            direction="lower",
        ),
        PreconditionSpec(
            name="harm_exposure_relative_deviation_bounded",
            description=(
                "This SHIFT arm's mean harm exposure must not deviate from the "
                "STATIONARY REFERENCE arm's by more than the ceiling. The level "
                "channel is only a MATCHED control for the PE channel if harm "
                "exposure really is matched; unmatched exposure would let C3 read an "
                "exposure difference as a channel dissociation. Measured, never "
                "assumed. Denominated on the REFERENCE arm and not on a pooled mean "
                "that includes this arm: with two arms a self-inclusive pooled "
                "deviation is |A-B|/(A+B), which is IDENTICAL for both arms and so "
                "can never single one out, and a 0.35 ceiling on it would admit a "
                "2.08x between-arm exposure ratio. Against the reference the same "
                "number means what it says -- a 1.25x ratio at the ceiling."),
            control=(
                "CEILING, pre-registered before the run at a 1.25x exposure ratio "
                "against the stationary reference; not derived from this run's own "
                "spread."),
            threshold=HARM_EXPOSURE_REL_DEV_MAX,
            direction="upper",
            applies_to=lambda ctx: bool(ctx.get("is_shift")),
            applies_note=(
                "Scoped OUT of ARM_0_STATIONARY: that arm IS the reference the "
                "deviation is measured against, so the precondition is not "
                "meaningful for it (disposition (a), not a vacuous arm)."),
        ),
        PreconditionSpec(
            name="pe_load_elevated_vs_stationary",
            description=(
                "The IV must actually have raised the harm-PE load in this SHIFT arm "
                "relative to the stationary reference. Below floor means the "
                "world_rule_shift manipulation never moved the channel C3 routes on, "
                "which is substrate-not-ready, NOT evidence about MECH-055."),
            control=(
                "V3-EXQ-861e's ecological_novelty_mel_gradient_present_this_config "
                "precondition was MET on this same IV (and its "
                "C1_measured_mel_gradient_present / C1_dv_spread_nonzero read true), "
                "so a graded response to the shift ladder is an established property "
                "of this lever rather than a hope."),
            threshold=PE_ELEVATION_MIN,
            direction="lower",
            applies_to=lambda ctx: bool(ctx.get("is_shift")),
            applies_note=(
                "Scoped OUT of ARM_0_STATIONARY: that arm IS the reference the "
                "elevation is measured against, so the precondition is not "
                "meaningful for it (disposition (a), not a vacuous arm)."),
        ),
        PreconditionSpec(
            name="fresh_select_sample_floor",
            description=(
                "Number of FRESH E3 selections in this arm's worst cell. The per-tick "
                "series are recorded only on fresh ticks (the E3 cadence latches "
                "diagnostics ~10x), so this is the TRUE denominator of every "
                "correlation below."),
            control=(
                "Step 2.5a probe, 2026-09-19, WITH the valence write paths driven as "
                "this run drives them: 592 fresh selections per 600 env steps (98.7%). "
                "So the 1800-step P2 budget projects to ~1780 fresh selections per "
                "cell. (An earlier probe of the same config WITHOUT those write paths "
                "measured only 29/220; the yield is config-dependent, which is exactly "
                "why this is a measured precondition and not an assumption.)"),
            threshold=float(FRESH_TICKS_MIN),
            direction="lower",
        ),
    ]


def _run(dry_run: bool):
    t0 = time.perf_counter()
    print("%s  dry_run=%s" % (EXPERIMENT_TYPE, dry_run))

    specs = _precondition_specs()
    arm_contexts = {a["arm_id"]: {"arm_id": a["arm_id"], "is_shift": a["is_shift"]}
                    for a in ARMS}
    # Design-time refusal: no arm may carry a precondition it provably cannot
    # meet. Runs BEFORE compute, and under --dry-run.
    assert_no_structurally_unsatisfiable_gate(specs, list(arm_contexts.values()))

    rows: List[Dict[str, Any]] = []
    seeds = SEEDS[:1] if dry_run else SEEDS
    for arm in ARMS:
        for seed in seeds:
            rows.append(_run_cell(arm, seed, dry_run))

    by_arm: Dict[str, List[Dict[str, Any]]] = {a["arm_id"]: [] for a in ARMS}
    for r in rows:
        by_arm[r["arm_id"]].append(r)

    # Pooled harm exposure, for the matched-control precondition.
    pooled_harm = _mean([r["mean_harm_exposure"] for r in rows])
    stationary_pe = _mean([r["mean_dacc_pe"] for r in by_arm[STATIONARY_ARM]])
    stationary_harm = _mean([r["mean_harm_exposure"] for r in by_arm[STATIONARY_ARM]])

    arm_gates = []
    for arm in ARMS:
        aid = arm["arm_id"]
        arows = by_arm[aid]
        r2_worst, r2_cell = _worst_cell(arows, "harm_a_forward_r2", "min")
        vh_worst, vh_cell = _worst_cell(arows, "range_valence_harm_delta", "min")
        pe_worst, pe_cell = _worst_cell(arows, "range_dacc_pe", "min")
        t_worst, t_cell = _worst_cell(arows, "range_axis1_temperature", "min")
        fresh_worst, fresh_cell = _worst_cell(arows, "n_fresh_select", "min")
        arm_harm = _mean([r["mean_harm_exposure"] for r in arows])
        arm_pe = _mean([r["mean_dacc_pe"] for r in arows])
        measured = {
            "harm_a_forward_r2_supra_floor": r2_worst,
            "valence_harm_series_range_supra_floor": vh_worst,
            "dacc_pe_series_range_supra_floor": pe_worst,
            "axis1_temperature_range_supra_floor": t_worst,
            "fresh_select_sample_floor": fresh_worst,
        }
        if arm["is_shift"]:
            measured["pe_load_elevated_vs_stationary"] = _rel_change(arm_pe, stationary_pe)
            measured["harm_exposure_relative_deviation_bounded"] = abs(
                _rel_change(arm_harm, stationary_harm))
        gate = evaluate_arm_gate(aid, arm_contexts[aid], specs, measured)
        for p in gate["preconditions"]:
            p["offending_cell"] = {
                "harm_a_forward_r2_supra_floor": r2_cell,
                "valence_harm_series_range_supra_floor": vh_cell,
                "dacc_pe_series_range_supra_floor": pe_cell,
                "axis1_temperature_range_supra_floor": t_cell,
                "fresh_select_sample_floor": fresh_cell,
            }.get(p["precondition"], aid)
        arm_gates.append(gate)

    agg = aggregate_arm_gates(arm_gates)
    green_arms = set(agg["green_arms"])

    # ------------------------------ criteria ------------------------------- #
    # Every criterion is scored on GREEN arms only -- a red arm's readouts are
    # not cited in either direction, and a red arm does NOT vacate a green one.
    def _quorum(aid: str, key: str) -> Tuple[int, int, bool]:
        arows = by_arm[aid]
        n_pass = sum(1 for r in arows if r[key])
        need = min(SEED_QUORUM, len(arows)) if arows else 0
        return n_pass, len(arows), bool(arows) and n_pass >= need

    c1_arms = [a["arm_id"] for a in ARMS
               if a["is_shift"] and a["arm_id"] in green_arms]
    c2_arms = [a["arm_id"] for a in ARMS if a["arm_id"] in green_arms]

    c1_by_arm = {aid: _quorum(aid, "c1_pass") for aid in c1_arms}
    c2_by_arm = {aid: _quorum(aid, "c2_pass") for aid in c2_arms}
    c4_by_arm = {aid: _quorum(aid, "c4_pass") for aid in c2_arms}

    c1_pass = bool(c1_by_arm) and all(v[2] for v in c1_by_arm.values())
    c2_pass = bool(c2_by_arm) and all(v[2] for v in c2_by_arm.values())
    c4_pass = bool(c4_by_arm) and all(v[2] for v in c4_by_arm.values())

    # C3: per seed, the PE channel's relative rise across the ladder must
    # exceed the level channel's by REL_CHANGE_DELTA_MIN. Both legs are
    # RELATIVE changes, so the comparison is scale-free -- a raw-magnitude
    # comparison would survive a broadcast constant (V3-EXQ-604c).
    c3_rows: List[Dict[str, Any]] = []
    c3_ready = STATIONARY_ARM in green_arms and HIGH_ARM in green_arms
    for seed in seeds:
        ref = next((r for r in by_arm[STATIONARY_ARM] if r["seed"] == seed), None)
        hi = next((r for r in by_arm[HIGH_ARM] if r["seed"] == seed), None)
        if ref is None or hi is None:
            continue
        d_pe = _rel_change(hi["mean_dacc_pe"], ref["mean_dacc_pe"])
        d_vh = _rel_change(hi["mean_valence_harm_delta"], ref["mean_valence_harm_delta"])
        delta = (abs(d_pe) - abs(d_vh)) if (math.isfinite(d_pe) and math.isfinite(d_vh)) \
            else float("nan")
        c3_rows.append({
            "seed": seed,
            "rel_change_dacc_pe": d_pe,
            "rel_change_valence_harm": d_vh,
            "differential": delta,
            "passed": bool(math.isfinite(delta) and delta >= REL_CHANGE_DELTA_MIN),
        })
    c3_n_pass = sum(1 for r in c3_rows if r["passed"])
    c3_decidable = bool(c3_ready and c3_rows)
    c3_pass = bool(c3_decidable and c3_n_pass >= min(SEED_QUORUM, len(c3_rows)))

    # ------------------------- non-degeneracy ------------------------------ #
    def _nd_series(keys: List[str]) -> bool:
        return all(
            any(math.isfinite(r[k]) and r[k] > 0.0 for r in rows) for k in keys
        ) and any(r["n_series"] >= CORR_MIN_N for r in rows)

    criteria_non_degenerate = {
        "C1_harm_channels_not_redundant": bool(
            _nd_series(["range_valence_harm_delta", "range_dacc_pe"])),
        "C2_axes_not_fixed_ratio": bool(
            _nd_series(["range_axis1_temperature",
                        "range_axis2_valence_spread_delta",
                        "range_dacc_pe"])),
        "C3_differential_channel_response": bool(
            c3_ready and _series_range(
                [r["mean_dacc_pe"] for r in rows]) > 0.0),
        "C4_timing_signatures_differ": bool(
            _nd_series(["range_valence_harm_delta", "range_dacc_pe"])),
    }

    # ---------------------------- verdict grid ----------------------------- #
    # DECIDABILITY IS CHECKED BEFORE ANY CLAIM VERDICT, and this ordering is
    # load-bearing rather than tidy. `c1_pass` is False both when the criterion
    # was evaluated and FAILED and when it could not be evaluated at all --
    # `c1_arms` is empty whenever no SHIFT arm passed its readiness gate, and
    # `c1_pass = bool(c1_by_arm) and ...` is then False. Without this guard a
    # shift arm going red (on `pe_load_elevated_vs_stationary`, whose own spec
    # text says a failure there "is substrate-not-ready, NOT evidence about
    # MECH-055") while the stationary arm stayed green would be recorded as
    # "MECH-055 FALSIFIER (ii) FIRED", evidence_direction `weakens` -- a
    # readiness failure laundered into a claim falsification, with nothing in
    # the manifest distinguishing it from a genuine one. Note that
    # aggregate_arm_gates' `non_degenerate` is ANY-arm-green by design
    # (precondition_gate.py: a red arm must not vacate a green one), so it
    # cannot carry this check on its own. Found by the Step 4.5 red-team pass.
    gate_green = bool(agg["non_degenerate"])
    undecidable = []
    if not gate_green:
        undecidable.append("no arm passed its readiness gate")
    if not c1_arms:
        undecidable.append(
            "C1 (the collapse-risk falsifier) is scored on SHIFT arms and no SHIFT "
            "arm passed its readiness gate")
    if not c2_arms:
        undecidable.append("C2 has no readiness-green arm to score")
    if not c3_decidable:
        undecidable.append(
            "C3 needs BOTH the stationary reference and the high-shift arm green, "
            "with at least one seed present in each")
    if undecidable:
        outcome = "FAIL"
        direction = "non_contributory"
        label = "substrate_not_ready_requeue"
        interp_note = (
            "NOT DECIDABLE on this run -- %s. Gate detail: %s. Re-queue under a NEW "
            "letter at an adequate P0. This is NOT a substrate ceiling, NOT a "
            "lockstep finding, and NOT a refutation of MECH-055; no criterion below "
            "should be read as a verdict."
            % ("; ".join(undecidable), agg["degeneracy_reason"] or "all arms green"))
    elif not c1_pass:
        outcome = "FAIL"
        direction = "weakens"
        label = "harm_channels_numerically_redundant_collapsed_scalar"
        interp_note = (
            "MECH-055 FALSIFIER (ii) FIRED: VALENCE_HARM_DISCRIMINATIVE and the "
            "dACC precision-weighted harm-forward PE are numerically redundant "
            "(|Spearman| > %.2f) in the decoupled SHIFT arm(s), i.e. one collapsed "
            "harm scalar wearing two labels -- the exact risk the claim's own "
            "COLLAPSE-RISK CHECK names, and the same failure class MECH-048 hit "
            "before its 2026-07-21 fix." % RHO_MAX)
    elif not c2_pass:
        outcome = "FAIL"
        direction = "weakens"
        label = "axes_move_in_fixed_ratio_lockstep"
        interp_note = (
            "MECH-055 FALSIFIER (i) FIRED: at least two of the measured axes move in "
            "a fixed, predictable ratio (pairwise R^2 > %.2f) despite the two harm "
            "representations being individually non-redundant." % R2_MAX)
    elif not c3_pass:
        outcome = "FAIL"
        direction = "mixed"
        label = "separation_present_but_manipulation_nonspecific"
        interp_note = (
            "Neither of MECH-055's falsifiers fired -- the channels are neither "
            "numerically redundant (C1) nor in fixed ratio (C2) -- but the decoupling "
            "manipulation did NOT move the PE channel measurably more than the level "
            "channel (C3), so this run does not establish the claim's "
            "'measurably different ... signatures UNDER a manipulation that decouples "
            "their upstream sources' clause. The separation evidence stands; the "
            "attribution to a decoupling manipulation does not.")
    else:
        outcome = "PASS"
        direction = "mixed"
        label = "two_axis_plus_harm_only_separation_supported_full_claim_awaits_benefit_channel"
        interp_note = (
            "The NARROWED two-axis-plus-harm-only test that MECH-055's own "
            "what_would_answer sanctions: both of the claim's falsifiers failed to "
            "fire (C1, C2) and the two harm representations responded differentially "
            "to a decoupling manipulation (C3). Direction is 'mixed', NOT 'supports': "
            "the claim as WORDED requires the harm/benefit duality inside axis 3, "
            "whose benefit half is architecturally absent (MECH-054, 2026-08-08), and "
            "the downstream-role half of the claim's first CONFIRMING clause is out of "
            "scope here (V3-EXQ-799's write_gate consumer gap). A full verdict on "
            "MECH-055 remains blocked on the benefit-side signed-PE channel.")

    run_id = "%s_%sZ_v3" % (EXPERIMENT_TYPE, datetime.utcnow().strftime("%Y%m%dT%H%M%S"))

    criteria = [
        {
            "name": "C1_harm_channels_not_redundant",
            "load_bearing": True,
            "passed": c1_pass,
            "measured": _worst_cell(
                [r for r in rows if r["arm_id"] in c1_arms],
                "abs_spearman_valence_harm_vs_dacc_pe", "max")[0],
            "threshold": RHO_MAX,
            "direction": "upper",
            "statistic": "max over scored cells of |Spearman(VALENCE_HARM_DISCRIMINATIVE, dacc_pe)|",
            "offending_cell": _worst_cell(
                [r for r in rows if r["arm_id"] in c1_arms],
                "abs_spearman_valence_harm_vs_dacc_pe", "max")[1],
            "scored_arms": c1_arms,
            "per_arm_seed_quorum": {k: {"n_pass": v[0], "n_seeds": v[1], "passed": v[2]}
                                    for k, v in c1_by_arm.items()},
            "maps_to": "MECH-055 FALSIFYING (ii)",
        },
        {
            "name": "C2_axes_not_fixed_ratio",
            "load_bearing": True,
            "passed": c2_pass,
            "measured": _worst_cell(
                [r for r in rows if r["arm_id"] in c2_arms],
                "max_pairwise_axis_r2", "max")[0],
            "threshold": R2_MAX,
            "direction": "upper",
            "statistic": "max over SCORED (readiness-green) cells and axis pairs of OLS R^2",
            "offending_cell": _worst_cell(
                [r for r in rows if r["arm_id"] in c2_arms],
                "max_pairwise_axis_r2", "max")[1],
            "scored_arms": c2_arms,
            "per_arm_seed_quorum": {k: {"n_pass": v[0], "n_seeds": v[1], "passed": v[2]}
                                    for k, v in c2_by_arm.items()},
            "maps_to": "MECH-055 FALSIFYING (i)",
        },
        {
            "name": "C3_differential_channel_response",
            "load_bearing": False,
            "passed": c3_pass,
            "measured": min((r["differential"] for r in c3_rows
                             if math.isfinite(r["differential"])), default=float("nan")),
            "threshold": REL_CHANGE_DELTA_MIN,
            "direction": "lower",
            "statistic": ("min over seeds of |rel_change(mean dacc_pe)| - "
                          "|rel_change(mean VALENCE_HARM_DISCRIMINATIVE)| across the "
                          "stationary -> high-shift ladder"),
            "n_pass": c3_n_pass,
            "n_seeds": len(c3_rows),
            "per_seed": c3_rows,
            "maps_to": "MECH-055 CONFIRMING, second clause",
        },
        {
            "name": "C4_timing_signatures_differ",
            "load_bearing": False,
            "passed": c4_pass,
            "measured": _worst_cell(
                [r for r in rows if r["arm_id"] in c2_arms],
                "lag1_abs_diff_pe_vs_valence_harm", "min")[0],
            "threshold": LAG1_DELTA_MIN,
            "direction": "lower",
            "statistic": ("min over SCORED (readiness-green) cells of "
                          "|lag1_autocorr(dacc_pe) - "
                          "lag1_autocorr(VALENCE_HARM_DISCRIMINATIVE increment)|"),
            "offending_cell": _worst_cell(
                [r for r in rows if r["arm_id"] in c2_arms],
                "lag1_abs_diff_pe_vs_valence_harm", "min")[1],
            "per_arm_seed_quorum": {k: {"n_pass": v[0], "n_seeds": v[1], "passed": v[2]}
                                    for k, v in c4_by_arm.items()},
            "maps_to": "MECH-055 CONFIRMING, second clause (order-sensitive companion)",
        },
    ]

    full_config = {
        "env": dict(ENV_KWARGS),
        "arms": ARMS,
        "world_rule_shift_depth": WORLD_RULE_SHIFT_DEPTH,
        "schedule": {
            "p0_eps": P0_EPS, "p1_eps": P1_EPS,
            "total_training_eps": TOTAL_TRAINING_EPS,
            "steps_per_episode": STEPS_PER_EPISODE,
            "p2_step_budget": P2_STEP_BUDGET,
            "epsilon_train": EPSILON_TRAIN, "epsilon_eval": EPSILON_EVAL,
        },
        "substrate": _config_slice(0)["substrate"],
        "thresholds": {
            "RHO_MAX": RHO_MAX, "R2_MAX": R2_MAX,
            "REL_CHANGE_DELTA_MIN": REL_CHANGE_DELTA_MIN,
            "LAG1_DELTA_MIN": LAG1_DELTA_MIN, "SEED_QUORUM": SEED_QUORUM,
            "FORWARD_R2_MIN": FORWARD_R2_MIN, "VH_RANGE_MIN": VH_RANGE_MIN,
            "TEMP_RANGE_MIN": TEMP_RANGE_MIN, "PE_RANGE_MIN": PE_RANGE_MIN,
            "HARM_EXPOSURE_REL_DEV_MAX": HARM_EXPOSURE_REL_DEV_MAX,
            "PE_ELEVATION_MIN": PE_ELEVATION_MIN,
            "FRESH_TICKS_MIN": FRESH_TICKS_MIN, "CORR_MIN_N": CORR_MIN_N,
        },
        "dry_run": dry_run,
    }

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "backlog_id": BACKLOG_ID,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "timestamp_utc": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "outcome": outcome,
        "evidence_direction": direction,
        "evidence_direction_per_claim": {"MECH-055": direction},
        "evidence_direction_note": interp_note,
        "sleep_driver_pattern": "none",
        "interpretation": {
            "label": label,
            "note": interp_note,
            "preconditions": agg["adjudication_preconditions"],
            "criteria_non_degenerate": criteria_non_degenerate,
        },
        "criteria": criteria,
        "combination_rule": (
            "PASS requires BOTH load-bearing criteria (C1 and C2) AND C3. C4 is "
            "corroborating and does not gate. Each criterion must hold in at least "
            "%d of the seeds of EVERY arm it is scored on; C1 is scored on the SHIFT "
            "arms only (the decoupling manipulation is applied there; the stationary "
            "arm's value is recorded as the reference), C2 and C4 on every green arm. "
            "Only arms whose readiness gate is green are scored." % SEED_QUORUM
        ),
        "per_arm_gate": agg["per_arm_gate"],
        "failed_preconditions_by_arm": agg.get("failed_preconditions_by_arm", {}),
        "arm_results": rows,
        "per_seed_results": rows,
        "c3_per_seed": c3_rows,
        "scope_note": (
            "NARROWED test, licensed by MECH-055's own what_would_answer. NOT in "
            "scope: (a) the benefit-side signed-PE channel, architecturally absent "
            "per MECH-054 2026-08-08, so no result here verdicts the harm/benefit "
            "duality the claim's wording requires; (b) the downstream/behavioural "
            "form of the claim's first CONFIRMING clause, blocked by V3-EXQ-799's "
            "write_gate consumer gap; (c) MECH-035's ranking claim -- the "
            "cross-candidate valence range is RECORDED here, not tested, and MECH-035 "
            "is deliberately NOT tagged."
        ),
        "dv_symmetry_note": (
            "DV family: Spearman correlations, OLS R^2, lag-1 autocorrelations and "
            "RELATIVE changes over per-tick series. Symmetry group: independent "
            "positive affine rescaling of either series, plus a uniform additive "
            "constant (exact for correlation/R^2; rescaling only for a relative "
            "change). The SAME statement holds for all three arms, which differ only "
            "in the rate of one manipulation: re-permuting the action -> displacement "
            "map REORDERS and RE-CONTENTS the z_harm_a sequence and destroys its "
            "action-conditioned predictability -- neither an affine rescaling nor an "
            "added constant of any measured series. So no arm's DV is invariant under "
            "its own manipulation and no arm is disposition-(b) vacuous. Designed "
            "around two consequences of that group: C3 compares RELATIVE changes "
            "rather than raw magnitudes because a broadcast constant survives a "
            "magnitude readout (V3-EXQ-604c); and because a pooled-over-ticks "
            "correlation IS permutation-invariant, C4's lag-1 autocorrelation is "
            "included as the order-SENSITIVE companion statistic."
        ),
        "red_team_note": {
            "step": "4.5 adversarial design review, one pass, model: fable",
            "verdict": "CONTESTED",
            "dispositions": [
                "F1 (verdict grid) FIXED -- an empty c1_arms made c1_pass False "
                "whether C1 was evaluated-and-failed or could not be evaluated at "
                "all, so a SHIFT arm going red on readiness while the stationary arm "
                "stayed green was recorded as 'MECH-055 FALSIFIER (ii) FIRED', "
                "direction weakens. aggregate_arm_gates' non_degenerate is "
                "any-arm-green by design and cannot carry the check. Now a "
                "decidability test runs BEFORE any claim verdict and routes to "
                "substrate_not_ready_requeue / non_contributory, naming which "
                "criterion was undecidable.",
                "F2 (manipulation cannot reach the DV) FIXED -- P1 trained and the "
                "P2 r2 estimator scored e2_harm_a(z(t-1), a(t)), pairing a transition "
                "with the action that had not caused it, and with a DIFFERENT pairing "
                "from the one the measured PE uses. Verified in ree_core: sense() "
                "caches _harm_a_prev = z(t) (agent.py:5583), select_action rolls "
                "pred = e2_harm_a(z(t), a(t)) (agent.py:10329), the next tick's dACC "
                "compares z(t+1) against it (agent.py:7743). Since world_rule_shift "
                "acts ONLY through the action channel, this blunted the very "
                "manipulation the design depends on. Both sites now use prev_action.",
                "F3 (gate certifying its own subject) FIXED -- agent.reset() clears "
                "_harm_a_pred_prev (agent.py:3630), so the first fresh tick of every "
                "episode reaches the dACC with z_harm_a_pred=None and "
                "dacc._affective_pe returns ||z_harm_a||, a LEVEL (dacc.py:213-214). "
                "Shift arms end episodes earlier, so mean_dacc_pe -- which both "
                "pe_load_elevated_vs_stationary and C3 read -- could have risen on "
                "episode COUNT alone. Those ticks are now excluded and counted in "
                "n_level_pe_ticks_excluded, and differencing is done WITHIN episodes.",
                "F4 (gate certifying its own subject) FIXED -- the matched-exposure "
                "gate was denominated on a pooled mean INCLUDING the arm under test. "
                "With two arms that statistic is |A-B|/(A+B), identical for both arms, "
                "so it could never single one out, and the 0.35 ceiling admitted a "
                "2.08x exposure ratio. Now denominated on the stationary REFERENCE "
                "arm, scoped out of that arm via applies_to, and tightened to 0.25 "
                "(a 1.25x ratio).",
                "F5 (criterion cannot discriminate) ALREADY ADDRESSED before the pass "
                "returned -- the residue-field integrator makes the level a ramp, so "
                "lag1 ~ 1 and C4 would pass trivially. Every criterion already routes "
                "on within-episode INCREMENTS, with levels recorded unrouted. The "
                "sub-claim that the sense() z_harm.norm() write path 'dominates' the "
                "channel is DISMISSED as not-a-defect: VALENCE_HARM_DISCRIMINATIVE is "
                "DEFINED as the z_harm_s-driven sensory-discriminative channel "
                "(residue/field.py:60, agent.py:5507), so its dominance is the channel "
                "behaving as specified, not contamination.",
                "F6 (manipulation cannot reach the DV, magnitude unverified) FIXED as "
                "INSTRUMENTATION -- the precision leg may be negligible rather than "
                "saturated (prec_norm = precision/5000; e3_selector.py cites "
                "current_precision ~95, giving a ~1.02 multiplier). The driver "
                "previously recorded only SATURATION, which would read 0 in that "
                "regime and be mistaken for 'precision is live'. It now records "
                "mean_prec_norm and the note covers BOTH inert regimes. Deliberately "
                "NOT gated: every routed statistic is invariant under a positive "
                "constant scaling, so an inert precision leg bounds what a PASS may "
                "claim about precision, but cannot change any verdict.",
                "F7 (verdict grid, minor) FIXED -- C1/C2/C4 headline measured and "
                "offending_cell values were taken over ALL rows, so a readiness-red "
                "arm's value could be reported as a criterion's number. Now scoped to "
                "the readiness-green arms each criterion is actually scored on.",
            ],
        },
        "precision_saturation_note": (
            "frac_precision_weight_saturated records the fraction of scored ticks on "
            "which dacc._affective_pe's precision term hit its cap "
            "(prec_norm = min(precision / dacc_precision_scale, 3.0), scale 5000 "
            "following V3-EXQ-597b). Above the cap the precision leg is a CONSTANT "
            "multiplier, so axis 3 is an unweighted forward-model residual times 4. "
            "This is RECORDED, not gated, and the reason is specific: every routed "
            "statistic here -- Spearman, OLS R^2, lag-1 autocorrelation, relative "
            "change -- is invariant under a positive constant scaling, so saturation "
            "can neither manufacture nor destroy any criterion's verdict. What it DOES "
            "bound is the reach of a PASS: if this fraction is near 1, the run has "
            "tested the separation of a forward-model RESIDUAL channel from the level "
            "channel, and has NOT exercised the precision-weighting that MECH-055's "
            "axis-3 wording also names. Read it before citing a PASS as evidence about "
            "precision specifically."
        ),
        "accumulation_note": (
            "Every routed statistic is computed on INSTANTANEOUS quantities. The "
            "residue-field channels ACCUMULATE (residue/field.py:294 -- update_valence "
            "adds rather than replaces, and every write path here is non-negative), so "
            "VALENCE_HARM_DISCRIMINATIVE and the valence spread are monotone ramps "
            "(Step 2.5a probe: 0 -> 460.4 over 592 ticks). C1 on a ramp-vs-residual "
            "pair would pass because a ramp and a fluctuating residual are nearly "
            "uncorrelated, and C4 would pass because a ramp's lag-1 autocorrelation is "
            "~1 -- both for reasons unrelated to channel separation, and both "
            "invisible in the manifest. C1/C2/C3/C4 therefore route on the per-tick "
            "INCREMENT of the accumulating channels, aligned tick-for-tick with the "
            "already-instantaneous temperature and harm-PE series. The levels are "
            "recorded but never routed, and "
            "abs_spearman_valence_harm_LEVEL_vs_dacc_pe_unrouted is C1's own statistic "
            "computed the WRONG way, kept so the size of the removed artifact is "
            "visible rather than asserted."
        ),
        "readiness_scope_note": (
            "Each readiness precondition certifies ONE channel and speaks for no "
            "other: harm_a_forward_r2 certifies axis 3 only; "
            "valence_harm_series_range certifies the VALENCE_HARM_DISCRIMINATIVE "
            "INCREMENT series only (and says nothing about its accumulating level, "
            "which is recorded but never routed); axis1_temperature_range certifies "
            "axis 1 only; "
            "harm_exposure_relative_deviation certifies the matched-control premise "
            "of C3 only."
        ),
        "levers_ruled_out_note": (
            "Three candidate decoupling levers were rejected on substrate evidence "
            "BEFORE this design, each recorded so a successor does not retry them: "
            "(1) SD-021/AIC descending attenuation of z_harm_s -- gated on the "
            "commitment latch via mode_weight = p_external * (beta_gate_elevated), "
            "aic_analog.py:249, so it cannot fire commitment-free; (2) "
            "harm_nonredundancy_weight -- V3-EXQ-323 measured baseline cosine_sq at "
            "8.5e-05 to 0.025, a floor with no headroom, and 323 failed its own C1 in "
            "2/5 seeds because of it (this knob is also a dataclass field with no "
            "from_dims kwarg, so from_dims silently swallows it); (3) "
            "env_drift_interval -- a measured null, V3-EXQ-677 produced a "
            "high-vs-low mean-PE difference of 8.8e-07 against a 0.01 threshold, and "
            "causal_grid_world.py's own SD-MEL-PRODUCER note explains why."
        ),
        "recorded_non_gating_note": (
            "cross_candidate_valence_range_post_action_mean is RECORDED, not gated: "
            "MECH-055's collapse-risk check does not depend on cross-candidate "
            "spread, so a monostrategy-degenerate candidate pool must not vacate C1. "
            "cross_candidate_valence_range_shared_seed_mean is its NEGATIVE CONTROL: "
            "world_states[0] is the rollout's shared initial z_world seed, "
            "bit-identical across candidates (E2FastPredictor.rollout_with_world), so "
            "a NON-zero shared-seed range would mean the range statistic itself is "
            "mis-implemented. HONEST LIMIT, measured up front: this experiment's Step "
            "2.5a probe measured BOTH reads at exactly 0 over 592 fresh ticks -- the "
            "candidates' rolled-out z_world lands far enough from every active RBF "
            "center that evaluate_valence returns ~0 for all of them, the collapsed- "
            "proposer regime config.py's candidate_summary_source note describes and "
            "V3-EXQ-614e measured (cand_world_pairwise_dist=0.0). So while the "
            "post-action read is 0 the negative control is UNINFORMATIVE -- it cannot "
            "distinguish a correct statistic from a broken one -- and neither number "
            "may be cited as evidence about MECH-035 or about candidate diversity. "
            "They are recorded only so a successor that fixes the proposer regime can "
            "see what this substrate did. This is precisely why neither gates: "
            "MECH-055's collapse-risk check reads the REALIZED z_world, not the "
            "candidate pool, so a degenerate pool must not vacate C1."
        ),
        "readout": flat_readout({
            "C1_harm_channels_not_redundant": c1_pass,
            "C2_axes_not_fixed_ratio": c2_pass,
            "C3_differential_channel_response": c3_pass,
            "C4_timing_signatures_differ": c4_pass,
            "overall_pass_flag": outcome == "PASS",
            "readiness_any_arm_green": gate_green,
            "n_green_arms": len(green_arms),
            "n_arms": len(ARMS),
            "n_criteria_passed": sum(1 for x in (c1_pass, c2_pass, c3_pass, c4_pass) if x),
            "n_criteria_total": 4,
            "rho_max": RHO_MAX,
            "r2_max": R2_MAX,
            "rel_change_delta_min": REL_CHANGE_DELTA_MIN,
            "lag1_delta_min": LAG1_DELTA_MIN,
            "forward_r2_min": FORWARD_R2_MIN,
            "abs_spearman_vh_pe_worst": _worst_cell(
                rows, "abs_spearman_valence_harm_vs_dacc_pe", "max")[0],
            "max_pairwise_axis_r2_worst": _worst_cell(
                rows, "max_pairwise_axis_r2", "max")[0],
            "lag1_abs_diff_worst": _worst_cell(
                rows, "lag1_abs_diff_pe_vs_valence_harm", "min")[0],
            "c3_differential_worst": min(
                (r["differential"] for r in c3_rows
                 if math.isfinite(r["differential"])), default=None),
            "harm_a_forward_r2_worst": _worst_cell(rows, "harm_a_forward_r2", "min")[0],
            "valence_harm_delta_range_worst": _worst_cell(
                rows, "range_valence_harm_delta", "min")[0],
            "valence_harm_level_range_worst": _worst_cell(
                rows, "range_valence_harm_level", "min")[0],
            "dacc_pe_range_worst": _worst_cell(rows, "range_dacc_pe", "min")[0],
            "axis1_temperature_range_worst": _worst_cell(
                rows, "range_axis1_temperature", "min")[0],
            "n_fresh_select_worst": _worst_cell(rows, "n_fresh_select", "min")[0],
            "n_latched_ticks_total": sum(r["n_latched_ticks"] for r in rows),
            "mean_dacc_pe_stationary": stationary_pe,
            "mean_dacc_pe_high_shift": _mean(
                [r["mean_dacc_pe"] for r in by_arm[HIGH_ARM]]),
            "mean_valence_harm_delta_stationary": _mean(
                [r["mean_valence_harm_delta"] for r in by_arm[STATIONARY_ARM]]),
            "mean_valence_harm_delta_high_shift": _mean(
                [r["mean_valence_harm_delta"] for r in by_arm[HIGH_ARM]]),
            "pooled_mean_harm_exposure": pooled_harm,
            "cross_candidate_range_post_action_mean": _mean(
                [r["cross_candidate_valence_range_post_action_mean"] for r in rows]),
            "cross_candidate_range_shared_seed_mean": _mean(
                [r["cross_candidate_valence_range_shared_seed_mean"] for r in rows]),
            "frac_precision_weight_saturated_mean": _mean(
                [r["frac_precision_weight_saturated"] for r in rows]),
            "n_cells": len(rows),
            "n_seeds": len(seeds),
        }),
        "custom_information": {
            "gov_reuse_1_check": (
                "Decisive readout = the JOINT per-tick pairing of "
                "VALENCE_HARM_DISCRIMINATIVE with the dACC precision-weighted "
                "harm-forward PE (and the three-axis lockstep R^2 over the same "
                "ticks). Checked via reanalysis_query.py over 1050 recorded "
                "manifests: node_valence_matrix is carried by exactly three runs "
                "(v3_exq_887 / 887a / 887b), and every one of them ran with "
                "use_dacc=False, so none carries a dACC PE to pair it with; the 11 "
                "dacc_pe carriers carry no valence store. "
                "v3_exq_876a_mech025_doing_mode_convergence_redesign appears in both "
                "keyword sets but only as CONFIG keys, with use_dacc=False and "
                "valence_harm_enabled=False. The pairing is therefore neither "
                "recorded nor derivable post-hoc, and the run additionally needs a "
                "manipulation (world_rule_shift as a harm-PE decoupler) present in no "
                "recorded MECH-055-relevant run. Not recoverable -> run."
            ),
            "step_2_5a_probe_2026_09_19": (
                "Empirical wiring probe on this exact config, before authoring, in "
                "two passes. PASS 1 (220 steps, valence write paths NOT driven): "
                "e2_harm_a built, dACC + adapter + salience coordinator + PCC all "
                "present, 32 candidates carrying world_states, env obs dims measured "
                "(body 12, world 250, action_dim 5 -- NOT the 4 some sibling drivers "
                "hardcode); but evaluate_valence returned ALL ZEROS, because no RBF "
                "center is active until update_residue runs and evaluate_valence "
                "short-circuits on active_mask.any(). PASS 2 (600 steps, write paths "
                "driven exactly as _drive_valence_write_paths drives them): "
                "VALENCE_HARM_DISCRIMINATIVE range 460.4, bundle['pe'] range 0.771, "
                "effective_temperature range 0.310, pcc_stability 0.0855-0.498, "
                "surprise_write_count 600, and 592 FRESH E3 selections per 600 env "
                "steps (98.7%, versus 29/220 in pass 1 -- the yield is "
                "config-dependent, which is why it is a measured precondition). "
                "Pass 2 is why _drive_valence_write_paths exists and why the "
                "valence_harm_series_range precondition gates. Both passes measured "
                "the cross-candidate valence range at EXACTLY 0 (see "
                "recorded_non_gating_note). Also confirmed from_dims returns "
                "harm_nonredundancy_weight=0.0 when passed, i.e. silently swallowed. "
                "Per-step cost measured at ~0.84 s CPU, which is what sized the "
                "schedule and dropped the third arm."
            ),
            "re_derive_brake": (
                "Counted 0 braking autopsies for MECH-055 (and 0 for MECH-035 and "
                "MECH-054) over the live failure_autopsy corpus at authoring time. "
                "Not braked."
            ),
        },
    }
    manifest["non_degenerate"] = bool(agg["non_degenerate"])
    if agg.get("degeneracy_reason"):
        manifest["degeneracy_reason"] = agg["degeneracy_reason"]

    out_path = write_flat_manifest(
        manifest,
        dry_run=dry_run,
        config=full_config,
        seeds=seeds,
        script_path=Path(__file__),
        started_at=t0,
        z_goal_stream_stats=_ZG.stats(),
    )
    print("  outcome=%s label=%s" % (outcome, label))
    print("  manifest -> %s" % out_path)
    return outcome, out_path, run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    _outcome, _out_path, _run_id = _run(args.dry_run)

    emit_outcome(
        outcome=_outcome if _outcome in ("PASS", "FAIL") else "FAIL",
        manifest_path=_out_path,
        run_id=_run_id,
        queue_id=QUEUE_ID,
        dry_run=args.dry_run,
    )
