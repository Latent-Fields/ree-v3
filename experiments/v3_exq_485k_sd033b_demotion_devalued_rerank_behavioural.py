#!/opt/local/bin/python3
"""V3-EXQ-485k: CORRECTED SD-033b trained-OFC-head BEHAVIOURAL DEVALUATION arm
(MECH-263 sig-a + sig-b) ON the MECH-448 (ARC-107) rank-preserving F->eligibility
DEMOTION-ENABLED E3 selector, with the demotion ENVELOPE FLOOR CALIBRATED per
(arm, seed) to the behavioural candidate bank.

commitment_closure:GAP-8 evidence-grade follow-on. SUPERSEDES V3-EXQ-485j (run_id
v3_exq_485j_sd033b_demotion_envelope_calibrated_behavioural_20260621T180033Z_v3),
which was a SPLIT result (failure_autopsy_V3-EXQ-485j_2026-06-21):

  C2 (task-role discrimination) CONVERTED -- between_context_selection_tv 1.0 on 2/3
    seeds; ARM_1 demotion-off F-dominance control = 0.0 all seeds; ARM_0 frozen-head
    silent. FIRST cross-substrate corroboration that the MECH-448 BG F->eligibility
    demotion lever GENERALISES off the GAP-A foraging substrate to the OFC valuation
    channel. MECH-448 is NOT in question (excluded_count=5, winner!=F-argmin -- the
    envelope fired correctly).

  C1 (devaluation) FAILED -- and the FAIL is C1-ONLY. The OFC trained head's bias
    cross-candidate RANGE collapses to ~0.02 (< the 0.05 DEVAL_SHIFT_MARGIN floor) AT
    THE DEVALUED STATE because the 485h/485j paired driver applied a DEVALUED_ANTIRANGE
    variance penalty (weight 0.5) that explicitly trains the head FLAT-at-devalued
    (low range) rather than RE-RANKING. A flat OFC bias at the devalued state gives the
    correctly-firing demotion selector (excluded_count=5, winner!=F-argmin) NOTHING to
    re-rank -> committed argmin unchanged -> devaluation_selection_shift = 0. This is a
    channel-specific SD-033b devalued-head / test-design gap, ORTHOGONAL to the
    f_eligibility_envelope_floor / dn_sigma (which fired correctly) -- so do NOT
    recalibrate the demotion envelope.

THE TWO 485k FIXES (off the 485j harness; the demotion lever + envelope calibration
are KEPT verbatim):

  (1) DEVALUED-STATE RE-RANKING DRIVER (replaces the anti-range collapse). At the
      devalued state the paired head driver now produces a RE-RANKING differentiated
      bias instead of a flat (anti-range) one: the high-threat outcome-coupling
      (favour low predicted harm) is INVERTED at the devalued state (favour high
      predicted harm) via the SAME REINFORCE-over-candidates form, so the head learns
      to DOWN-bias the previously-preferred candidate / UP-bias an alternative -- the
      mammalian outcome-devaluation re-ranking signature (Rudebeck & Murray 2014;
      Dickinson & Balleine). The devalued bias now carries genuine cross-candidate
      range, so devaluation ACTIVELY MOVES the committed selection through the demotion
      selector. This is NON-VACUOUS: the driver only gives the head a differentiated
      devalued bias; whether the committed argmin actually shifts still depends on the
      MECH-448 F-eligible set + within-eligible arbitration. ADDITIONALLY a secondary
      bias-VECTOR readout (high-vs-devalued L2 + cosine shift) is recorded so the
      autopsy can read the valuation change even if the committed-argmax TV shift is
      thin.

  (2) C1 READINESS PRECONDITION RE-TARGETED to the DEVALUED-STATE bias range. 485j
      gated the readiness precondition on the HIGH-THREAT range (~851/_bias_range at
      the high-threat state) while the C1 DV routes on the DEVALUED-state range
      (~880), so a below-floor devalued arm masqueraded as a fair `weakens` (the
      V3-EXQ-642 same-statistic miss). 485k adds bias_range_devalued > BIAS_RANGE_FLOOR
      to the ARM_2 readiness gate (alongside the high-threat range = C2 positive
      control + head-trained + MECH-448 non-degeneracy), so a below-floor devalued
      range now self-routes substrate_not_ready_requeue (non_contributory), NEVER a
      false SD-033b/MECH-263 weakens.

KEPT FROM 485j (do NOT change): ofc_bias_scale=0.5; SD-056 e2_action_contrastive armed
  in P0; phased training (P0 encoder warmup with SD-056 contrastive -> P1 head training
  on the frozen-encoder state_code -> P2 BEHAVIOURAL eval); the OFC-isolated SD-054
  bipartite reef/forage env (ofc_harm_dim>0); LR_OFC_BIAS 2e-3 / P1 120 ep; the
  MECH-448 demotion-on config (use_f_eligibility_demotion / dn_sigma=0.0) + the
  per-(arm,seed) f_eligibility_envelope_floor CALIBRATION (485i/485j fix); the
  excluded_count>0 MECH-448 non-degeneracy HARD gate; and the 3-arm dissociation.

THREE ARMS (clean dissociation; conversion attributable to the CONJUNCTION
trained-head AND demotion -- not either alone):
  ARM_0_frozen_demotion_on   : frozen zeroed OFC head + demotion ON. Silence control.
  ARM_1_trained_demotion_off : trained OFC head + demotion OFF. F-dominance ceiling
                               control (the trained valuation reaches the accumulator
                               but F dominates the argmin -> ~0 conversion).
  ARM_2_trained_demotion_on  : trained OFC head + demotion ON. The TEST -- trained
                               valuation + F demoted to eligibility -> the OFC channel
                               arbitrates the committed action within the F-eligible
                               set with F removed from the final argmin.
  All three pass the OFC bias as score_bias and use the SAME settle/eval pipeline; the
  ONLY varied factors are (head trained?) x (demotion on?).

PRE-REGISTERED ACCEPTANCE (per seed, then >= MIN_PASS_SEEDS of 3):
  READINESS (non-vacuity precondition, on the ARM_2 test arm; SAME statistics the
    load-bearing DVs route on):
      (a) trained-head OFC bias cross-candidate RANGE at the HIGH-THREAT positive
          control > BIAS_RANGE_FLOOR (0.05) -- the C2 discrimination positive control.
      (b) trained-head OFC bias cross-candidate RANGE at the DEVALUED state >
          BIAS_RANGE_FLOOR (0.05) -- the 485k FIX 2: the SAME statistic the
          load-bearing C1 devaluation_selection_shift routes on. The re-ranking
          devalued driver must produce a differentiated (non-flat) devalued bias.
      (c) head trained (state_bias_head weight-delta > HEAD_DELTA_MIN).
      (d) MECH-448 NON-DEGENERACY (kept from 485j as a HARD gate): the calibrated
          envelope actually EXCLUDED a candidate (f_eligibility_excluded_count > 0 AND
          f_eligibility_demotion_active), not an all-admit no-op.
    Any below floor / unmet -> substrate_not_ready_requeue (non_contributory), NEVER a
    weakens.
  C1_devaluation_behavioural_shift (load-bearing): ARM_2 devaluation_selection_shift
    > DEVAL_SHIFT_MARGIN AND > ARM_1 (demotion-off ceiling) shift + DEVAL_SHIFT_MARGIN
    -- conversion attributable to the demotion lever, not the trained head alone.
  C2_discrimination_behavioural_separation (load-bearing): ARM_2 separation_ratio
    >= SEPARATION_RATIO_MIN AND between_context_selection_tv >= BETWEEN_TV_FLOOR AND
    > ARM_1 between_tv + BETWEEN_TV_FLOOR.
  C3_silence_control (load-bearing): ARM_0 (frozen head, demotion on) shift < 1e-9
    AND between-context TV < 1e-9.
  PASS (supports SD-033b + MECH-263) = READINESS AND C1 AND C2 AND C3 on a majority.

INTERPRETATION GRID (one row per plausible outcome -> next action):
  PASS / supports: the trained OFC outcome-value channel, routed through the
    demotion-enabled selector, produces devaluation-sensitive AND task-role-
    discriminative committed selection. SD-033b candidate -> provisional behavioural
    evidence; contributes to closing behavioral_diversity_isolation:GAP-I (both
    signatures converted).
  FAIL / weakens (readiness MET, DVs fail): the trained OFC channel has genuine
    cross-candidate range at BOTH the high-threat and devalued states AND the demotion
    envelope genuinely excluded F-dominant candidates, yet committed selection still
    does not convert. Route /failure-autopsy (conversion-ceiling-persists-despite-
    demotion -> MECH-449 follow-on / V4) BEFORE stamping. An honest result.
  FAIL / substrate_not_ready_requeue (readiness UNMET): below the 0.05 bias-range
    floor at the high-threat OR DEVALUED state, head untrained, OR the calibrated
    envelope still admitted all (excluded_count == 0) so the demotion never engaged ->
    the behavioural test never ran. Re-queue with a stronger / different devalued
    driver. NOT a weakens.

SLEEP DRIVER: not applicable (no sleep loop in this experiment).
"""

from __future__ import annotations

import argparse
import json
import random
import socket
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome
from experiments._lib.arm_fingerprint import arm_cell
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_485k_sd033b_demotion_devalued_rerank_behavioural"
EXPERIMENT_PURPOSE = "evidence"
QUEUE_ID = "V3-EXQ-485k"
CLAIM_IDS: List[str] = ["SD-033b", "MECH-263"]

SELF_DIM = 8
WORLD_DIM = 32
HARM_DIM = 4
HARM_A_DIM = 4
HARM_HISTORY_LEN = 10

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

P0_EPISODES = 30
P1_EPISODES = 120                      # kept from 485h: give the driver budget
STEPS_PER_EPISODE = 100
TRAIN_EPS = P0_EPISODES + P1_EPISODES  # progress denominator M

LR_E1 = 1e-3
LR_E2_WF = 1e-3
LR_E3_HARM = 1e-3
LR_OFC_BIAS = 2e-3                     # kept from 485h
BATCH_SIZE = 32
WF_BUF_MAX = 256
N_PROBE_CANDIDATES = 8
RECORD_EVERY_N_STEPS = 4
OUTCOME_BUF_MAX = 512
POLICY_TEMPERATURE = 1.0
EMA_DECAY = 0.9

# -- 485h paired threat-conditioned driver (high-threat spread kept verbatim) --
OUTCOME_COUPLE_GAIN = 4.0
ADV_MIN_THRESHOLD = 1e-4
# -- 485k FIX 1: the 485h/485j DEVALUED_ANTIRANGE variance penalty (which trained the
#    devalued-state head FLAT -> nothing for the demotion selector to re-rank -> C1
#    shift=0) is REPLACED by a RE-RANKING driver. At the devalued state the high-threat
#    outcome-coupling is INVERTED (favour high predicted harm) via the SAME
#    REINFORCE-over-candidates form, so devaluation DOWN-biases the previously-preferred
#    candidate / UP-biases an alternative (Rudebeck & Murray 2014; Dickinson & Balleine)
#    -> a differentiated devalued bias the demotion selector can re-rank.
DEVALUED_RERANK_WEIGHT = 0.5   # weight of the devalued re-ranking REINFORCE term
DEVALUED_RERANK_GAIN = 4.0     # outcome-coupling gain at the devalued state (threat ~0
                               # there, so a constant gain replaces the threat-norm scale)
DRIVER_SNAP_BATCH = 16

# -- 485f Fix 1 (kept): defeat OFC clamp-saturation --
OFC_BIAS_SCALE = 0.5

# -- 485f Fix 2 (kept): SD-056 e2_action_contrastive in P0 -> action-divergent bank --
E2_CONTRASTIVE_WEIGHT = 0.05
E2_CONTRASTIVE_TEMP = 0.1

# -- P2 behavioural-eval drive lengths (parallel to 485b/485c/485f/485g/485h) --
PRE_ONSET_TICKS = 25
POST_ONSET_TICKS = 40
CONTEXT_TICKS = 25

# -- MECH-448 (ARC-107) demotion lever constants (matched on the demotion-on arms) --
# F_ELIG_ENVELOPE_FLOOR is the INITIAL (substrate-default) floor; the 485j fix
# CALIBRATES it per (arm, seed) to the behavioural bank's measured merit-share
# distribution at eval (see _calibrate_envelope_floor) so the envelope genuinely
# EXCLUDES a non-empty tail (the 485i excluded_count==0 all-admit no-op fix).
F_ELIG_ENVELOPE_FLOOR = 0.30   # absolute DN-share floor (substrate default / fallback)
F_ELIG_DN_SIGMA = 0.0          # DN semi-saturation / global tightness (KEEP 0.0)
# 485j envelope-floor calibration: keep the F-best top-k (small enough that the OFC
# channel arbitrates a real subset, per the 569i small-rotating-set conversion lesson),
# excluding a non-empty tail. Calibration picks the largest share-gap among cut points
# in [KEEP_MIN, KEEP_MAX]; no clean gap (flat/near-flat F) -> calibration FAILS -> floor
# stays at the default -> envelope all-admits -> precond-3 (excluded_count>0) FAILS.
ENVELOPE_KEEP_MIN = 2          # eligible set must keep >= 2 so the OFC bias can arbitrate
ENVELOPE_KEEP_MAX = 4          # ... and <= 4 of 8 so the eligible set stays small

# -- Deterministic committed readout: force the argmin-within-eligible path so each
#    DV is the genuine committed selection of the demotion-enabled selector (not a
#    multinomial sample). Eval-time only; training keeps commitment_threshold=0.5. --
EVAL_COMMIT_THRESHOLD = 1.0e6

# -- Pre-registered thresholds (NOT derived from the run's own statistics) --
HEAD_DELTA_MIN = 1e-3
BIAS_RANGE_FLOOR = 0.05      # == DEVAL_SHIFT_MARGIN (same-statistic non-vacuity gate)
DEVAL_SHIFT_MARGIN = 0.05
SEPARATION_RATIO_MIN = 3.0
BETWEEN_TV_FLOOR = 0.05
FROZEN_SILENCE_EPS = 1e-9
MIN_PASS_SEEDS = 2


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _action_to_onehot(idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
    return v


def _onehot_vec(idx: int, k: int) -> torch.Tensor:
    """[K] one-hot of the committed candidate index (the deterministic committed
    selection distribution under the demotion selector)."""
    v = torch.zeros(k)
    if 0 <= idx < k:
        v[idx] = 1.0
    return v


def _tv(p: torch.Tensor, q: torch.Tensor) -> float:
    """Total-variation distance between two selection distributions."""
    return float(0.5 * (p - q).abs().sum().item())


def _bias_range(agent: REEAgent, bank: torch.Tensor) -> float:
    """Cross-candidate range (max - min) of the OFC bias over the bank -- the
    readiness statistic the load-bearing selection DVs route on."""
    bias = agent.ofc.compute_bias(bank).detach()
    if bias.numel() < 2:
        return 0.0
    return float((bias.max() - bias.min()).item())


def _bank_zworld_spread(bank: torch.Tensor) -> float:
    """Mean pairwise L2 across the K candidate-bank z_world summaries (diagnostic)."""
    if bank is None or bank.shape[0] < 2:
        return 0.0
    d = torch.cdist(bank, bank)
    K = bank.shape[0]
    eye = torch.eye(K, dtype=torch.bool, device=bank.device)
    return float(d[~eye].mean().item())


def _calibrate_envelope_floor(
    raw_scores: torch.Tensor,
    dn_sigma: float = F_ELIG_DN_SIGMA,
    keep_min: int = ENVELOPE_KEEP_MIN,
    keep_max: int = ENVELOPE_KEEP_MAX,
) -> Optional[float]:
    """485j fix for the 485i excluded_count==0 all-admit no-op.

    The MECH-448 envelope (e3_selector._f_eligibility_envelope) admits candidate i iff
    its merit-share elig[i] = merit[i] / (dn_sigma + sum(merit)) clears an ABSOLUTE
    floor (merit[i] = max(F) - F[i]; F = raw_scores, a per-candidate cost). On the
    OFC-isolated behavioural bank the F pool is SPREAD, so with the 689d-tuned
    floor=0.30 no candidate's share cleared it and the envelope fell back to all-admit
    (excluded_count==0) -> demotion no-op.

    This CALIBRATES an ABSOLUTE floor to the bank's actual merit-share distribution:
    keep the F-best top-k (k in [keep_min, keep_max]) and exclude the rest, choosing
    the cut with the LARGEST share-gap so the eligible/excluded boundary is the most
    natural (and the eligible set stays small, per the 569i small-rotating-set
    conversion lesson). The floor is the midpoint of that gap -- still an ABSOLUTE
    share value (NOT a fraction of max). raw_scores is bias-independent and constant
    across the eval drives within a seed, so one calibration governs all drives.

    Returns the calibrated floor, or None when no clean gap exists in
    [keep_min, keep_max] (flat / near-flat / single-outlier-only F where the top
    shares are tied) -- the caller then leaves the default floor so the envelope
    all-admits and precond-3 (excluded_count>0) correctly FAILS to
    substrate_not_ready_requeue. dn_sigma MUST match the agent config value (0.0).
    """
    n = int(raw_scores.shape[0])
    if n < 2:
        return None
    merit = (raw_scores.max() - raw_scores).clamp(min=0.0)
    merit_sum = float(merit.sum().item())
    if merit_sum <= 1e-8:
        return None  # flat F -- no discrimination, no calibratable floor
    elig = (merit / (dn_sigma + merit_sum)).detach().cpu()
    sorted_desc, _ = torch.sort(elig, descending=True)
    k_hi = min(keep_max, n - 1)          # must exclude >= 1 candidate
    best_gap = -1.0
    best_floor: Optional[float] = None
    for keep in range(keep_min, k_hi + 1):
        hi = float(sorted_desc[keep - 1].item())   # smallest kept share
        lo = float(sorted_desc[keep].item())       # largest excluded share
        gap = hi - lo
        if gap > 1e-6 and gap > best_gap:
            best_gap = gap
            best_floor = 0.5 * (hi + lo)
    return best_floor


def _settle_state_code(
    agent: REEAgent,
    z_world_rep: torch.Tensor,
    z_harm: Optional[torch.Tensor],
    ticks: int,
    reset: bool,
) -> None:
    """Drive the OFC state_code toward a regime via gated EMA updates. SHARED by the
    P1 driver and the P2 eval so the trained and tested state_codes are constructed
    identically (kept from 485h)."""
    if reset:
        agent.ofc.reset()
    for _ in range(ticks):
        agent.ofc.update(z_world=z_world_rep, z_harm=z_harm, gate=1.0)


def _make_agent(seed: int, train_head: bool, demotion: bool) -> Tuple[REEAgent, CausalGridWorldV2]:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    env = CausalGridWorldV2(seed=seed, **ENV_KWARGS)
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        harm_dim=HARM_DIM,
        alpha_world=0.9,
        alpha_self=0.3,
        reafference_action_dim=env.action_dim,
        use_harm_stream=True,
        z_harm_dim=HARM_DIM,
        use_affective_harm_stream=True,
        z_harm_a_dim=HARM_A_DIM,
        harm_history_len=HARM_HISTORY_LEN,
        use_resource_proximity_head=True,
        resource_proximity_weight=0.5,
        benefit_eval_enabled=True,
        benefit_weight=1.0,
        z_goal_enabled=True,
        goal_weight=0.5,
        drive_weight=2.0,
        e1_goal_conditioned=True,
        # OFC isolated: no gated_policy / dACC / lateral_pfc bias channels.
        use_gated_policy=False,
        use_dacc=False,
        dacc_weight=0.0,
        use_lateral_pfc_analog=False,
        # SP-CEM main-path for first-action candidate diversity.
        use_support_preserving_cem=True,
        support_preserving_stratified_elites=True,
        support_preserving_ao_std_floor=0.2,
        support_preserving_min_first_action_classes=2,
        # 485f Fix 2 (kept): SD-056 e2 action-conditional contrastive.
        e2_action_contrastive_enabled=True,
        e2_action_contrastive_weight=E2_CONTRASTIVE_WEIGHT,
        e2_action_contrastive_temperature=E2_CONTRASTIVE_TEMP,
        e2_action_contrastive_min_batch_classes=2,
        # SD-033b OFC analog under test. ofc_harm_dim>0 -> z_harm enters state_code.
        use_ofc_analog=True,
        ofc_state_dim=16,
        ofc_harm_dim=HARM_DIM,
        ofc_bias_scale=OFC_BIAS_SCALE,
        ofc_train_state_bias_head=train_head,
        # MECH-448 (ARC-107) rank-preserving F->eligibility demotion (the ARM_ON
        # variable). The OFC bias is passed as score_bias at eval -> _modulatory_accum
        # is non-None -> the demotion eligibility branch arbitrates the committed
        # action within the F-eligible set with F removed from the final argmin.
        use_f_eligibility_demotion=demotion,
        f_eligibility_envelope_floor=F_ELIG_ENVELOPE_FLOOR,
        f_eligibility_dn_sigma=F_ELIG_DN_SIGMA,
    )
    config.e3.commitment_threshold = 0.5
    config.heartbeat.beta_gate_bistable = True
    agent = REEAgent(config)
    return agent, env


def _preflight() -> None:
    frozen, _ = _make_agent(1, train_head=False, demotion=True)
    trainable, _ = _make_agent(2, train_head=True, demotion=True)
    ceiling, _ = _make_agent(3, train_head=True, demotion=False)
    assert frozen.ofc is not None and trainable.ofc is not None
    assert frozen.ofc.config.train_state_bias_head is False
    assert trainable.ofc.config.train_state_bias_head is True
    assert frozen.ofc.config.harm_dim == HARM_DIM
    assert abs(float(trainable.ofc.config.bias_scale) - OFC_BIAS_SCALE) < 1e-9
    assert hasattr(trainable.e2, "world_forward_contrastive_loss")
    assert bool(getattr(trainable.config.e2, "e2_action_contrastive_enabled", False))
    # readiness floor re-aligned to the DV floor (same-statistic gate).
    assert abs(BIAS_RANGE_FLOOR - DEVAL_SHIFT_MARGIN) < 1e-9
    fz = frozen.ofc.state_bias_head[-1]
    tz = trainable.ofc.state_bias_head[-1]
    assert bool(torch.all(fz.weight == 0)) and bool(torch.all(fz.bias == 0))
    assert not (bool(torch.all(tz.weight == 0)) and bool(torch.all(tz.bias == 0)))
    assert len(list(trainable.ofc.bias_head_parameters())) == 4
    # MECH-448 contract: E3Config carries the demotion lever flags; ARM_ON True.
    assert hasattr(trainable.e3.config, "use_f_eligibility_demotion")
    assert trainable.e3.config.use_f_eligibility_demotion is True
    assert ceiling.e3.config.use_f_eligibility_demotion is False
    assert abs(float(trainable.e3.config.f_eligibility_envelope_floor) - F_ELIG_ENVELOPE_FLOOR) < 1e-9
    del frozen, trainable, ceiling
    print(
        "Preflight PASS: OFC analog + harm_dim + trainable/frozen head + "
        "bias_head_parameters + ofc_bias_scale=0.5 + SD-056 contrastive + readiness "
        "floor 0.05 == DV floor + paired threat-conditioned driver + MECH-448 "
        "demotion flags wired (ARM_ON True / ARM_OFF False)",
        flush=True,
    )


def _build_snap(agent: REEAgent, candidates: List) -> Optional[torch.Tensor]:
    """First-step z_world summaries [K, world_dim] from the proposer candidates."""
    if not isinstance(candidates, list) or len(candidates) < N_PROBE_CANDIDATES:
        return None
    if getattr(candidates[0], "world_states", None) is None:
        return None
    if len(candidates[0].world_states) < 2:
        return None
    return torch.cat(
        [c.world_states[1].detach().clone() for c in candidates[:N_PROBE_CANDIDATES]],
        dim=0,
    )  # [K, world_dim]


def _head_weight_vector(agent: REEAgent) -> torch.Tensor:
    return torch.cat(
        [p.detach().reshape(-1).cpu() for p in agent.ofc.bias_head_parameters()]
    )


# --------------------------------------------------------------------------- #
# training (P0 encoder warmup + P1 paired threat-conditioned head driver)
# kept verbatim from 485h -- the demotion flag does NOT affect training, only the
# eval committed selection through E3.select().
# --------------------------------------------------------------------------- #
def _encoder_step(agent: REEAgent, env, steps: int, arm: str, ep_global: int) -> float:
    device = agent.device
    e1_opt = optim.Adam(agent.e1.parameters(), lr=LR_E1)
    e2_opt = optim.Adam(
        list(agent.e2.world_transition.parameters())
        + list(agent.e2.world_action_encoder.parameters()),
        lr=LR_E2_WF,
    )
    he_opt = optim.Adam(agent.e3.harm_eval_head.parameters(), lr=LR_E3_HARM)
    wf_buf: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    he_buf: List[Tuple[torch.Tensor, torch.Tensor]] = []
    ep_reward = 0.0
    z_wp = z_sp = act_p = None
    _, obs_dict = env.reset()
    agent.reset()
    for _ in range(steps):
        obs_body = obs_dict["body_state"]
        obs_world = obs_dict["world_state"]
        latent = agent.sense(
            obs_body,
            obs_world,
            obs_harm=obs_dict.get("harm_obs"),
            obs_harm_a=obs_dict.get("harm_obs_a"),
            obs_harm_history=obs_dict.get("harm_history"),
        )
        z_w = latent.z_world.detach()
        if z_wp is not None and act_p is not None:
            agent.record_transition(z_sp, act_p, latent.z_self.detach())
        ticks = agent.clock.advance()
        e1_prior = (
            agent._e1_tick(latent)
            if ticks.get("e1_tick", False)
            else torch.zeros(1, WORLD_DIM, device=device)
        )
        candidates = agent.generate_trajectories(latent, e1_prior, ticks)
        drive = REEAgent.compute_drive_level(obs_body)
        agent.update_z_goal(
            benefit_exposure=max(0.0, float(obs_dict.get("benefit_exposure", 0.0))),
            drive_level=drive,
        )
        action = agent.select_action(candidates, ticks, temperature=1.0)
        if action is None:
            action = _action_to_onehot(
                random.randint(0, env.action_dim - 1), env.action_dim, device
            )
            agent._last_action = action
        _, harm, done, _, obs_dict = env.step(action)
        ep_reward += float(harm)
        if z_wp is not None and act_p is not None:
            wf_buf.append((z_wp.cpu(), act_p.cpu(), z_w.cpu()))
            if len(wf_buf) > WF_BUF_MAX:
                wf_buf = wf_buf[-WF_BUF_MAX:]
        he_buf.append(
            (z_w.cpu(), torch.tensor([abs(float(harm)) if float(harm) < 0 else 0.0]))
        )
        if len(wf_buf) >= BATCH_SIZE:
            idx = torch.randperm(len(wf_buf))[:BATCH_SIZE].tolist()
            zw = torch.cat([wf_buf[i][0] for i in idx]).to(device)
            a = torch.cat([wf_buf[i][1] for i in idx]).to(device)
            zw1 = torch.cat([wf_buf[i][2] for i in idx]).to(device)
            pred = agent.e2.world_forward(zw, a)
            loss = F.mse_loss(pred, zw1)
            # SD-056 action-contrastive auxiliary loss (kept from 485f/485g/485h).
            contrast = agent.e2.world_forward_contrastive_loss(
                zw, a, zw1,
                temperature=E2_CONTRASTIVE_TEMP,
                min_batch_classes=2,
            )
            loss = loss + E2_CONTRASTIVE_WEIGHT * contrast
            if loss.requires_grad:
                e2_opt.zero_grad()
                loss.backward()
                e2_opt.step()
            with torch.no_grad():
                agent.e3.update_running_variance((pred.detach() - zw1).detach())
        if len(he_buf) >= BATCH_SIZE:
            idx = torch.randperm(len(he_buf))[:BATCH_SIZE].tolist()
            zw = torch.cat([he_buf[i][0] for i in idx]).to(device)
            ht = torch.cat([he_buf[i][1] for i in idx]).to(device)
            loss = F.mse_loss(agent.e3.harm_eval(zw).squeeze(), ht.squeeze())
            if loss.requires_grad:
                he_opt.zero_grad()
                loss.backward()
                he_opt.step()
        if len(agent._world_experience_buffer) >= 2:
            e1_loss = agent.compute_prediction_loss()
            if e1_loss.requires_grad:
                e1_opt.zero_grad()
                e1_loss.backward()
                e1_opt.step()
        z_wp, z_sp, act_p = z_w, latent.z_self.detach(), action.detach()
        if done:
            break
    if ep_global == 1 or ep_global % 10 == 0:
        print(
            f"  [train] {arm} ep {ep_global}/{TRAIN_EPS}  phase=P0  reward={ep_reward:.3f}",
            flush=True,
        )
    return ep_reward


def _ofc_threat_conditioned_driver_loss(
    agent: REEAgent,
    snap_buf: List[Tuple[torch.Tensor, torch.Tensor]],
    z_harm_high: torch.Tensor,
    z_harm_low: torch.Tensor,
    device,
) -> Tuple[torch.Tensor, int, int]:
    """485k paired threat-conditioned, outcome-coupled, per-candidate driver. Trains the
    trained-OFC head on BOTH regimes using a SHARED state_code construction
    (_settle_state_code) identical to the eval:
      - high-threat regime (KEPT verbatim from 485h): spread bias favouring LOW
        predicted harm (REINFORCE over candidates, advantage = harm.mean() - harm).
      - devalued regime (485k FIX 1, RE-RANKING; REPLACES the 485h/485j anti-range
        variance penalty): the SAME REINFORCE form with the advantage INVERTED
        (harm - harm.mean()), so the devalued head learns to FAVOUR what the
        high-threat head disfavoured -- down-bias the previously-preferred candidate /
        up-bias an alternative (the mammalian outcome-devaluation re-ranking signature,
        Rudebeck & Murray 2014; Dickinson & Balleine). The devalued bias is now
        DIFFERENTIATED (carries cross-candidate range), so the demotion selector has a
        re-ranking to act on -> devaluation moves the committed selection (C1).
    Returns (loss, n_spread, n_rerank). Side effect: mutates ofc.state_code; the caller
    saves/restores it."""
    if agent.ofc is None or len(snap_buf) < 2:
        return torch.zeros(1, device=device), 0, 0
    threat_high = float(z_harm_high.detach().norm().item())
    idxs = np.random.choice(
        len(snap_buf), size=min(DRIVER_SNAP_BATCH, len(snap_buf)), replace=False
    )
    spread_terms: List[torch.Tensor] = []
    rerank_terms: List[torch.Tensor] = []
    for i in idxs:
        bank, z_world_rep = snap_buf[int(i)]
        if bank.shape[0] < 2:
            continue
        # per-candidate harm cost is state_code-INDEPENDENT (harm_eval reads the bank,
        # not the OFC state_code) -- compute once, reuse in both regimes.
        with torch.no_grad():
            harm = agent.e3.harm_eval(bank).reshape(-1)  # [K] per-candidate harm cost
        if harm.numel() != bank.shape[0]:
            continue
        # ---- high-threat regime: spread (favour low predicted harm) ----
        _settle_state_code(agent, z_world_rep, z_harm_high, PRE_ONSET_TICKS, reset=True)
        with torch.no_grad():
            adv_hi = OUTCOME_COUPLE_GAIN * threat_high * (harm.mean() - harm)  # [K]
        if float(adv_hi.abs().max()) >= ADV_MIN_THRESHOLD:
            bias_hi = agent.ofc.compute_bias(bank)  # [K], grad flows into the head
            log_p_hi = F.log_softmax(-bias_hi / POLICY_TEMPERATURE, dim=0)
            spread_terms.append(-(adv_hi * log_p_hi).sum())
        # ---- devalued regime: RE-RANK (inverted advantage; continue high->devalued) ----
        _settle_state_code(agent, z_world_rep, z_harm_low, POST_ONSET_TICKS, reset=False)
        with torch.no_grad():
            # INVERTED advantage: favour HIGH predicted harm at the devalued state, so
            # the devalued head re-ranks AGAINST the high-threat low-harm preference.
            adv_dev = DEVALUED_RERANK_GAIN * (harm - harm.mean())  # [K]
        if float(adv_dev.abs().max()) >= ADV_MIN_THRESHOLD:
            bias_dev = agent.ofc.compute_bias(bank)  # [K], grad flows into the head
            log_p_dev = F.log_softmax(-bias_dev / POLICY_TEMPERATURE, dim=0)
            rerank_terms.append(-(adv_dev * log_p_dev).sum())
    if not spread_terms and not rerank_terms:
        return torch.zeros(1, device=device), 0, 0
    loss = torch.zeros(1, device=device)
    if spread_terms:
        loss = loss + torch.stack(spread_terms).mean()
    if rerank_terms:
        loss = loss + DEVALUED_RERANK_WEIGHT * torch.stack(rerank_terms).mean()
    return loss, len(spread_terms), len(rerank_terms)


def _p1_train(
    agent: REEAgent,
    env: CausalGridWorldV2,
    episodes: int,
    steps: int,
    arm: str,
    train_head: bool,
    z_harm_high: torch.Tensor,
    z_harm_low: torch.Tensor,
) -> Dict:
    device = agent.device
    bias_opt = (
        optim.Adam(list(agent.ofc.bias_head_parameters()), lr=LR_OFC_BIAS)
        if train_head
        else None
    )
    head_init = _head_weight_vector(agent)
    grad_nonzero_updates = 0
    n_spread_terms = 0
    n_rerank_terms = 0
    snap_buf: List[Tuple[torch.Tensor, torch.Tensor]] = []
    bias_samples: List[float] = []
    agent.train()
    for ep in range(episodes):
        _, obs_dict = env.reset()
        agent.reset()
        step = 0
        z_wp = z_sp = act_p = None
        for _ in range(steps):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            latent = agent.sense(
                obs_body,
                obs_world,
                obs_harm=obs_dict.get("harm_obs"),
                obs_harm_a=obs_dict.get("harm_obs_a"),
                obs_harm_history=obs_dict.get("harm_history"),
            )
            if z_wp is not None and act_p is not None:
                agent.record_transition(z_sp, act_p, latent.z_self.detach())
            ticks = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent)
                if ticks.get("e1_tick", False)
                else torch.zeros(1, WORLD_DIM, device=device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            agent.update_z_goal(
                benefit_exposure=max(0.0, float(obs_dict.get("benefit_exposure", 0.0))),
                drive_level=REEAgent.compute_drive_level(obs_body),
            )
            if step % RECORD_EVERY_N_STEPS == 0:
                snap = _build_snap(agent, candidates)
                if snap is not None:
                    snap_buf.append((snap, latent.z_world.detach().clone()))
            action = agent.select_action(candidates, ticks, temperature=1.0)
            if action is None:
                action = _action_to_onehot(
                    random.randint(0, env.action_dim - 1), env.action_dim, device
                )
                agent._last_action = action
            if agent.ofc is not None:
                bias_samples.append(float(agent.ofc._last_bias_abs_mean))
            _, harm, done, _, obs_dict = env.step(action)
            z_wp, z_sp, act_p = (
                latent.z_world.detach(),
                latent.z_self.detach(),
                action.detach(),
            )
            step += 1
            if done:
                break
        if len(snap_buf) > OUTCOME_BUF_MAX:
            snap_buf = snap_buf[-OUTCOME_BUF_MAX:]
        if bias_opt is not None and len(snap_buf) >= 2:
            saved_state = agent.ofc.state_code.detach().clone()
            l_loss, n_sp, n_rr = _ofc_threat_conditioned_driver_loss(
                agent, snap_buf, z_harm_high, z_harm_low, device
            )
            with torch.no_grad():
                agent.ofc.state_code.copy_(saved_state)
            n_spread_terms += n_sp
            n_rerank_terms += n_rr
            if l_loss.requires_grad:
                bias_opt.zero_grad()
                l_loss.backward()
                gsum = sum(
                    float(p.grad.abs().sum())
                    for p in agent.ofc.bias_head_parameters()
                    if p.grad is not None
                )
                if gsum > 0:
                    grad_nonzero_updates += 1
                torch.nn.utils.clip_grad_norm_(agent.ofc.bias_head_parameters(), 1.0)
                bias_opt.step()
        ep_global = P0_EPISODES + ep + 1
        if ep_global % 10 == 0:
            print(
                f"  [train] {arm} ep {ep_global}/{TRAIN_EPS}  phase=P1"
                f"  bias_abs={bias_samples[-1] if bias_samples else 0:.5f}"
                f"  grad_updates={grad_nonzero_updates}",
                flush=True,
            )
    head_final = _head_weight_vector(agent)
    head_delta = float(torch.norm(head_final - head_init).item())
    return {
        "p1_mean_abs_ofc_bias": float(np.mean(bias_samples)) if bias_samples else 0.0,
        "head_weight_delta_norm": head_delta,
        "n_grad_nonzero_updates": grad_nonzero_updates,
        "n_spread_loss_terms": n_spread_terms,
        "n_rerank_loss_terms": n_rerank_terms,
    }


# --------------------------------------------------------------------------- #
# P2 behavioural eval on the demotion-enabled E3 selector
# --------------------------------------------------------------------------- #
def _collect_eval_candidates(
    agent: REEAgent, env: CausalGridWorldV2, steps: int
) -> Tuple[Optional[List], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Run the trained agent briefly and return (candidates[:K], candidate_bank
    [K, world_dim], z_world_rep [1, world_dim]) -- a real candidate list aligned with
    its first-step z_world bank for the controlled devaluation / discrimination drives
    (kept from 485h)."""
    device = agent.device
    cands: Optional[List] = None
    bank: Optional[torch.Tensor] = None
    z_world_rep: Optional[torch.Tensor] = None
    agent.eval()
    with torch.no_grad():
        _, obs_dict = env.reset()
        agent.reset()
        for _ in range(steps):
            latent = agent.sense(
                obs_dict["body_state"],
                obs_dict["world_state"],
                obs_harm=obs_dict.get("harm_obs"),
                obs_harm_a=obs_dict.get("harm_obs_a"),
                obs_harm_history=obs_dict.get("harm_history"),
            )
            ticks = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent)
                if ticks.get("e1_tick", False)
                else torch.zeros(1, WORLD_DIM, device=device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            snap = _build_snap(agent, candidates)
            if snap is not None and bank is None:
                bank = snap
                z_world_rep = latent.z_world.detach().clone()
                cands = list(candidates[:N_PROBE_CANDIDATES])
            action = agent.select_action(candidates, ticks, temperature=1.0)
            if action is None:
                action = _action_to_onehot(0, env.action_dim, device)
            _, _, done, _, obs_dict = env.step(action)
            if bank is not None:
                break
            if done:
                _, obs_dict = env.reset()
                agent.reset()
    return cands, bank, z_world_rep


def _committed_select(
    agent: REEAgent, cands: List, score_bias: torch.Tensor, temperature: float
) -> Tuple[int, bool, Dict, float]:
    """Route the OFC bias through the REAL E3 selector as score_bias and return the
    COMMITTED candidate index + commit flag + diagnostics + raw (F-score) range.

    With use_f_eligibility_demotion on the agent's E3Config, the OFC bias (passed as
    score_bias -> _modulatory_accum) arbitrates the committed action WITHIN the
    F-eligible envelope, F removed from the final argmin. EVAL_COMMIT_THRESHOLD has
    forced the committed (argmin-within-eligible) path so selected_index is
    deterministic -- the genuine committed selection of the demotion-enabled selector."""
    res = agent.e3.select(
        cands,
        temperature=temperature,
        goal_state=getattr(agent, "goal_state", None),
        score_bias=score_bias,
    )
    diag = dict(agent.e3.last_score_diagnostics)
    raw = agent.e3.last_raw_scores
    raw_range = (
        float((raw.max() - raw.min()).item())
        if raw is not None and raw.numel() > 1
        else 0.0
    )
    return int(res.selected_index), bool(res.committed), diag, raw_range


def _demotion_committed_eval(
    agent: REEAgent, cands: List, bank: torch.Tensor, z_world_rep: torch.Tensor,
    z_harm_high: torch.Tensor, z_harm_low: torch.Tensor, seed: int,
) -> Dict:
    """PRIMARY behavioural eval: devaluation-selection shift + task-role discrimination
    measured on the COMMITTED selection through the demotion-enabled E3 selector.
    Records the readiness positive control (OFC bias range at the high-threat state),
    the devalued-state range (the 485h disambiguator), and the MECH-448 non-degeneracy
    signals (f_eligibility_excluded_count / envelope_size / demotion_active)."""
    K = bank.shape[0]
    agent.eval()
    prev_thr = float(agent.e3.config.commitment_threshold)
    agent.e3.config.commitment_threshold = EVAL_COMMIT_THRESHOLD
    excluded_count = -1
    envelope_size = -1
    demotion_active = False
    winner_neq_f = False
    raw_range_hi = 0.0
    calibrated_floor = -1.0
    calibration_succeeded = False
    deval_bias_l2_shift = 0.0      # 485k secondary readout: ||bias_high - bias_low||
    deval_bias_cosine = 1.0        # 485k secondary readout: cos(bias_high, bias_low)
    try:
        with torch.no_grad():
            # ---- P2-a devaluation sensitivity ----
            _settle_state_code(agent, z_world_rep, z_harm_high, PRE_ONSET_TICKS, reset=True)
            bias_high = agent.ofc.compute_bias(bank).detach()
            bias_range_high = _bias_range(agent, bank)  # readiness positive control

            # 485j envelope-floor CALIBRATION (the 485i excluded_count==0 fix). On the
            # demotion-on arms, probe the primary F pool (raw_scores; bias-independent
            # and constant across the drives) over the real candidate set, then set an
            # ABSOLUTE f_eligibility_envelope_floor calibrated to the bank's merit-share
            # spread so the MECH-448 envelope EXCLUDES a non-empty F-worst tail. A flat /
            # near-flat F (no clean share-gap) leaves the default floor -> all-admit ->
            # excluded_count==0 -> precond-3 self-routes substrate_not_ready_requeue.
            if bool(agent.e3.config.use_f_eligibility_demotion):
                _committed_select(agent, cands, bias_high, POLICY_TEMPERATURE)  # raw probe
                _raw = agent.e3.last_raw_scores
                if _raw is not None and _raw.numel() >= 2:
                    _floor = _calibrate_envelope_floor(_raw.detach())
                    if _floor is not None:
                        agent.e3.config.f_eligibility_envelope_floor = float(_floor)
                        calibrated_floor = float(_floor)
                        calibration_succeeded = True

            idx_high, comm_high, diag_hi, raw_range_hi = _committed_select(
                agent, cands, bias_high, POLICY_TEMPERATURE
            )
            excluded_count = int(diag_hi.get("f_eligibility_excluded_count", -1))
            envelope_size = int(diag_hi.get("f_eligibility_envelope_size", -1))
            demotion_active = bool(diag_hi.get("f_eligibility_demotion_active", False))
            winner_neq_f = bool(diag_hi.get("f_eligibility_winner_neq_f_argmin", False))

            _settle_state_code(agent, z_world_rep, z_harm_low, POST_ONSET_TICKS, reset=False)
            bias_low = agent.ofc.compute_bias(bank).detach()
            bias_range_devalued = _bias_range(agent, bank)  # the C1 DV routes on this
            # 485k secondary readout: the high-vs-devalued OFC bias-VECTOR change. A
            # re-ranking devalued driver moves the bias vector (large L2 shift, low/
            # negative cosine) even when the committed-argmax TV shift is thin -- lets
            # the autopsy read the valuation change directly (the user's "additionally").
            _bh = bias_high.reshape(-1)
            _bl = bias_low.reshape(-1)
            deval_bias_l2_shift = float((_bh - _bl).norm().item())
            deval_bias_cosine = (
                float(F.cosine_similarity(_bh.unsqueeze(0), _bl.unsqueeze(0), dim=1).item())
                if _bh.numel() > 1
                else 1.0
            )
            idx_low, comm_low, _, _ = _committed_select(
                agent, cands, bias_low, POLICY_TEMPERATURE
            )
            deval_shift = _tv(_onehot_vec(idx_high, K), _onehot_vec(idx_low, K))

            # ---- P2-b task-role discrimination ----
            ctx = CONTEXT_TICKS
            g = torch.Generator().manual_seed(20000 + seed)
            hist_a = z_world_rep + 1.5 + 0.1 * torch.randn(ctx, WORLD_DIM, generator=g)
            hist_b = z_world_rep - 1.5 + 0.1 * torch.randn(ctx, WORLD_DIM, generator=g)
            hist_a2 = z_world_rep + 1.5 + 0.1 * torch.randn(ctx, WORLD_DIM, generator=g)

            def _drive_then_select(hist: torch.Tensor) -> Tuple[int, float]:
                agent.ofc.reset()
                for t in range(hist.shape[0]):
                    agent.ofc.update(z_world=hist[t:t + 1], z_harm=None, gate=1.0)
                agent.ofc.update(z_world=z_world_rep, z_harm=None, gate=1.0)
                b = agent.ofc.compute_bias(bank).detach()
                idx, _, _, _ = _committed_select(agent, cands, b, POLICY_TEMPERATURE)
                rng = _bias_range(agent, bank)
                return idx, rng

            idx_a, bias_range_context_a = _drive_then_select(hist_a)
            idx_b, bias_range_context_b = _drive_then_select(hist_b)
            idx_a2, _ = _drive_then_select(hist_a2)
            between = _tv(_onehot_vec(idx_a, K), _onehot_vec(idx_b, K))
            within = _tv(_onehot_vec(idx_a, K), _onehot_vec(idx_a2, K))
            separation_ratio = between / max(within, 1e-6)
    finally:
        agent.e3.config.commitment_threshold = prev_thr

    collapse_ratio = (
        bias_range_devalued / bias_range_high if bias_range_high > 1e-9 else 0.0
    )
    return {
        "devaluation_selection_shift": deval_shift,
        "bias_range_high_threat": bias_range_high,
        "bias_range_devalued": bias_range_devalued,
        "devaluation_bias_l2_shift": deval_bias_l2_shift,
        "devaluation_bias_cosine": deval_bias_cosine,
        "devaluation_range_collapse_ratio": collapse_ratio,
        "bias_range_context_a": bias_range_context_a,
        "bias_range_context_b": bias_range_context_b,
        "bank_zworld_spread": _bank_zworld_spread(bank),
        "between_context_selection_tv": between,
        "within_context_selection_jitter": within,
        "discrimination_separation_ratio": separation_ratio,
        "committed_high": comm_high,
        "committed_low": comm_low,
        "f_eligibility_demotion_active": demotion_active,
        "f_eligibility_excluded_count": excluded_count,
        "f_eligibility_envelope_size": envelope_size,
        "f_eligibility_winner_neq_f_argmin": winner_neq_f,
        "raw_score_range_high": raw_range_hi,
        "f_eligibility_envelope_floor_calibrated": calibrated_floor,
        "f_eligibility_calibration_succeeded": calibration_succeeded,
    }


def _empty_eval() -> Dict:
    return {
        "devaluation_selection_shift": 0.0,
        "bias_range_high_threat": 0.0,
        "bias_range_devalued": 0.0,
        "devaluation_bias_l2_shift": 0.0,
        "devaluation_bias_cosine": 1.0,
        "devaluation_range_collapse_ratio": 0.0,
        "bias_range_context_a": 0.0,
        "bias_range_context_b": 0.0,
        "bank_zworld_spread": 0.0,
        "between_context_selection_tv": 0.0,
        "within_context_selection_jitter": 0.0,
        "discrimination_separation_ratio": 0.0,
        "committed_high": False,
        "committed_low": False,
        "f_eligibility_demotion_active": False,
        "f_eligibility_excluded_count": -1,
        "f_eligibility_envelope_size": -1,
        "f_eligibility_winner_neq_f_argmin": False,
        "raw_score_range_high": 0.0,
        "f_eligibility_envelope_floor_calibrated": -1.0,
        "f_eligibility_calibration_succeeded": False,
        "eval_context_built": False,
    }


def run_arm_seed(arm: str, train_head: bool, demotion: bool, seed: int, dry_run: bool) -> Dict:
    p0 = 3 if dry_run else P0_EPISODES
    p1 = 4 if dry_run else P1_EPISODES
    steps = 30 if dry_run else STEPS_PER_EPISODE
    print(
        f"\nSeed {seed} Condition {arm} train_head={train_head} demotion={demotion}",
        flush=True,
    )
    full_config = {
        "arm": arm,
        "train_head": train_head,
        "demotion": demotion,
        "p0": p0,
        "p1": p1,
        "steps": steps,
        "ofc_bias_scale": OFC_BIAS_SCALE,
        "e2_contrastive_weight": E2_CONTRASTIVE_WEIGHT,
        "lr_ofc_bias": LR_OFC_BIAS,
        "outcome_couple_gain": OUTCOME_COUPLE_GAIN,
        "devalued_rerank_weight": DEVALUED_RERANK_WEIGHT,
        "devalued_rerank_gain": DEVALUED_RERANK_GAIN,
        "bias_range_floor": BIAS_RANGE_FLOOR,
        "f_eligibility_envelope_floor_default": F_ELIG_ENVELOPE_FLOOR,
        "f_eligibility_dn_sigma": F_ELIG_DN_SIGMA,
        "envelope_keep_min": ENVELOPE_KEEP_MIN,
        "envelope_keep_max": ENVELOPE_KEEP_MAX,
        "eval_commit_threshold": EVAL_COMMIT_THRESHOLD,
        "env_kwargs": ENV_KWARGS,
    }
    with arm_cell(seed, config_slice=full_config, script_path=Path(__file__)) as cell:
        agent, env = _make_agent(seed, train_head=train_head, demotion=demotion)
        # Canonical high-threat / devalued z_harm, built ONCE per (arm, seed) and
        # shared by the P1 driver and the P2 eval so trained regimes == tested.
        g = torch.Generator().manual_seed(20000 + seed)
        z_harm_high = torch.randn(1, HARM_DIM, generator=g) * 1.0
        z_harm_low = torch.zeros(1, HARM_DIM)
        for ep in range(p0):
            _encoder_step(agent, env, steps, arm, ep + 1)
        p1m = _p1_train(agent, env, p1, steps, arm, train_head, z_harm_high, z_harm_low)
        cands, bank, z_world_rep = _collect_eval_candidates(agent, env, steps)
        if bank is None or z_world_rep is None or cands is None:
            beh = _empty_eval()
        else:
            beh = _demotion_committed_eval(
                agent, cands, bank, z_world_rep, z_harm_high, z_harm_low, seed
            )
            beh["eval_context_built"] = True
        row = {
            "arm": arm,
            "seed": seed,
            "ofc_train_state_bias_head": train_head,
            "use_f_eligibility_demotion": demotion,
            "head_weight_delta_norm": p1m["head_weight_delta_norm"],
            "n_grad_nonzero_updates": p1m["n_grad_nonzero_updates"],
            "n_spread_loss_terms": p1m["n_spread_loss_terms"],
            "n_rerank_loss_terms": p1m["n_rerank_loss_terms"],
            "p1_mean_abs_ofc_bias": p1m["p1_mean_abs_ofc_bias"],
            **beh,
        }
        cell.stamp(row)
    print(
        f"verdict: {'PASS' if beh.get('eval_context_built') else 'FAIL'}  "
        f"head_delta={p1m['head_weight_delta_norm']:.6f}  "
        f"bias_range_hi={beh['bias_range_high_threat']:.6f}  "
        f"bias_range_dev={beh['bias_range_devalued']:.6f}  "
        f"deval_shift={beh['devaluation_selection_shift']:.4f}  "
        f"deval_bias_l2={beh['devaluation_bias_l2_shift']:.4f}  "
        f"between_tv={beh['between_context_selection_tv']:.4f}  "
        f"excluded={beh['f_eligibility_excluded_count']}  "
        f"demotion_active={beh['f_eligibility_demotion_active']}",
        flush=True,
    )
    return row


# --------------------------------------------------------------------------- #
# aggregation + interpretation
# --------------------------------------------------------------------------- #
ARM_FROZEN = "ARM_0_frozen_demotion_on"
ARM_CEILING = "ARM_1_trained_demotion_off"
ARM_TEST = "ARM_2_trained_demotion_on"


def run(seeds: Optional[List[int]] = None, dry_run: bool = False) -> Dict:
    if seeds is None:
        seeds = [0, 1, 2]
    arms = [
        (ARM_FROZEN, False, True),
        (ARM_CEILING, True, False),
        (ARM_TEST, True, True),
    ]
    by_arm: Dict[str, List[Dict]] = {a[0]: [] for a in arms}
    for seed in seeds:
        for arm_label, train_head, demotion in arms:
            by_arm[arm_label].append(
                run_arm_seed(arm_label, train_head, demotion, seed, dry_run)
            )

    frozen = by_arm[ARM_FROZEN]
    ceiling = by_arm[ARM_CEILING]
    test = by_arm[ARM_TEST]
    n = len(seeds)

    # READINESS (non-vacuity, on the ARM_2 test arm). 485k FIX 2: the gate now requires
    # the DEVALUED-state bias range too (the SAME statistic the load-bearing C1
    # devaluation_selection_shift routes on -- 485j gated only the high-threat range
    # while C1 routed on the devalued range, so a below-floor devalued arm masqueraded
    # as a fair weakens). A below-floor devalued range now self-routes
    # substrate_not_ready_requeue (non_contributory), NEVER a false weakens.
    ready_seeds = sum(
        1
        for r in test
        if r["head_weight_delta_norm"] > HEAD_DELTA_MIN
        and r["bias_range_high_threat"] > BIAS_RANGE_FLOOR
        and r["bias_range_devalued"] > BIAS_RANGE_FLOOR
        and r.get("eval_context_built", False)
    )
    ready_devalued_seeds = sum(
        1 for r in test if r["bias_range_devalued"] > BIAS_RANGE_FLOOR
    )
    nondegen_seeds = sum(
        1
        for r in test
        if int(r.get("f_eligibility_excluded_count", -1)) > 0
        and bool(r.get("f_eligibility_demotion_active", False))
    )
    readiness_met = ready_seeds >= MIN_PASS_SEEDS and nondegen_seeds >= MIN_PASS_SEEDS

    c1_seeds = 0
    c2_seeds = 0
    for ce, te in zip(ceiling, test):
        if (
            te["devaluation_selection_shift"] > DEVAL_SHIFT_MARGIN
            and te["devaluation_selection_shift"]
            > ce["devaluation_selection_shift"] + DEVAL_SHIFT_MARGIN
        ):
            c1_seeds += 1
        if (
            te["discrimination_separation_ratio"] >= SEPARATION_RATIO_MIN
            and te["between_context_selection_tv"] >= BETWEEN_TV_FLOOR
            and te["between_context_selection_tv"]
            > ce["between_context_selection_tv"] + BETWEEN_TV_FLOOR
        ):
            c2_seeds += 1
    c3_seeds = sum(
        1
        for r in frozen
        if r["devaluation_selection_shift"] < FROZEN_SILENCE_EPS
        and r["between_context_selection_tv"] < FROZEN_SILENCE_EPS
    )

    c1 = c1_seeds >= MIN_PASS_SEEDS
    c2 = c2_seeds >= MIN_PASS_SEEDS
    c3 = c3_seeds >= MIN_PASS_SEEDS
    overall_pass = readiness_met and c1 and c2 and c3

    return {
        "by_arm": by_arm,
        "n_seeds": n,
        "ready_seeds": ready_seeds,
        "ready_devalued_seeds": ready_devalued_seeds,
        "nondegen_seeds": nondegen_seeds,
        "readiness_met": readiness_met,
        "max_test_head_delta": max((r["head_weight_delta_norm"] for r in test), default=0.0),
        "max_test_bias_range": max((r["bias_range_high_threat"] for r in test), default=0.0),
        "max_test_bias_range_devalued": max((r["bias_range_devalued"] for r in test), default=0.0),
        "max_test_deval_bias_l2_shift": max((r["devaluation_bias_l2_shift"] for r in test), default=0.0),
        "min_test_deval_bias_cosine": min((r["devaluation_bias_cosine"] for r in test), default=1.0),
        "max_test_excluded_count": max((int(r.get("f_eligibility_excluded_count", -1)) for r in test), default=-1),
        "max_test_between_tv": max((r["between_context_selection_tv"] for r in test), default=0.0),
        "max_test_deval_shift": max((r["devaluation_selection_shift"] for r in test), default=0.0),
        "max_ceiling_deval_shift": max((r["devaluation_selection_shift"] for r in ceiling), default=0.0),
        "max_ceiling_between_tv": max((r["between_context_selection_tv"] for r in ceiling), default=0.0),
        "acceptance": {
            "C1_devaluation_behavioural_shift": c1,
            "C2_discrimination_behavioural_separation": c2,
            "C3_silence_control": c3,
            "n_c1_seeds": c1_seeds,
            "n_c2_seeds": c2_seeds,
            "n_c3_seeds": c3_seeds,
            "pass": overall_pass,
        },
    }


def _interpretation(result: Dict) -> Dict:
    acc = result["acceptance"]
    ready = bool(result["readiness_met"])
    if not ready:
        label = "substrate_not_ready_requeue"
    elif acc["pass"]:
        label = "sd033b_demotion_conversion_supported"
    else:
        label = "conversion_ceiling_persists_despite_demotion"

    preconditions = [
        {
            "name": "ofc_trained_head_bias_cross_candidate_range_supra_dv_floor_HIGH_THREAT",
            "description": (
                "ARM_2 trained-head OFC bias cross-candidate RANGE (max-min over the "
                "real candidate bank) at the HIGH-THREAT positive-control state clears "
                "BIAS_RANGE_FLOOR (0.05). The C2 discrimination positive control -- the "
                "head can produce cross-candidate range at all. RANGE not magnitude. "
                "Below floor -> substrate_not_ready_requeue, NEVER a weakens."
            ),
            "measured": float(result["max_test_bias_range"]),
            "threshold": BIAS_RANGE_FLOOR,
            "control": "ARM_2 trained-arm bias over a real candidate bank at the high-threat state",
            "met": result["max_test_bias_range"] > BIAS_RANGE_FLOOR,
        },
        {
            "name": "ofc_trained_head_bias_cross_candidate_range_supra_dv_floor_DEVALUED_STATE",
            "description": (
                "485k FIX 2: ARM_2 trained-head OFC bias cross-candidate RANGE at the "
                "DEVALUED state clears BIAS_RANGE_FLOOR (0.05) -- the SAME statistic the "
                "load-bearing C1 devaluation_selection_shift routes on. 485j gated only "
                "the HIGH-THREAT range while C1 routed on the devalued range, so a "
                "below-floor devalued arm masqueraded as a fair weakens (the V3-EXQ-642 "
                "same-statistic miss). The 485k re-ranking devalued driver must produce a "
                "DIFFERENTIATED (non-flat) devalued bias for the demotion selector to "
                "re-rank. Below floor -> substrate_not_ready_requeue, NEVER a weakens."
            ),
            "measured": float(result["max_test_bias_range_devalued"]),
            "threshold": BIAS_RANGE_FLOOR,
            "control": "ARM_2 trained-arm bias over a real candidate bank at the DEVALUED state",
            "met": result["max_test_bias_range_devalued"] > BIAS_RANGE_FLOOR,
        },
        {
            "name": "ofc_head_weight_delta_supra_floor",
            "description": (
                "ARM_2 state_bias_head weight-delta-from-init L2 norm clears "
                "HEAD_DELTA_MIN -- the head genuinely trained under the paired "
                "threat-conditioned driver."
            ),
            "measured": float(result["max_test_head_delta"]),
            "threshold": HEAD_DELTA_MIN,
            "control": "trainable-arm head trained on frozen-encoder state_code in P1",
            "met": result["max_test_head_delta"] > HEAD_DELTA_MIN,
        },
        {
            "name": "mech448_f_eligibility_excluded_count_supra_zero",
            "description": (
                "MECH-448 NON-DEGENERACY: on ARM_2 the F->eligibility envelope actually "
                "EXCLUDED at least one candidate (f_eligibility_excluded_count > 0) AND "
                "the demotion path was active -- F was genuinely demoted on a divergent "
                "F pool, not an all-admit no-op. excluded_count == 0 (non-divergent F) -> "
                "substrate_not_ready_requeue, NEVER a weakens."
            ),
            "measured": float(result["max_test_excluded_count"]),
            "threshold": 0.0,
            "control": "ARM_2 demotion-on E3.select diagnostics on the trained-agent eval bank",
            "met": result["nondegen_seeds"] >= MIN_PASS_SEEDS,
        },
    ]
    criteria_non_degenerate = {
        "C1": bool(acc["C1_devaluation_behavioural_shift"]) and ready,
        "C2": bool(acc["C2_discrimination_behavioural_separation"]) and ready,
        "C3": bool(acc["C3_silence_control"]),
    }
    criteria = [
        {"name": "C1_devaluation_behavioural_shift", "load_bearing": True, "passed": bool(acc["C1_devaluation_behavioural_shift"])},
        {"name": "C2_discrimination_behavioural_separation", "load_bearing": True, "passed": bool(acc["C2_discrimination_behavioural_separation"])},
        {"name": "C3_silence_control", "load_bearing": True, "passed": bool(acc["C3_silence_control"])},
    ]
    return {
        "label": label,
        "preconditions": preconditions,
        "criteria_non_degenerate": criteria_non_degenerate,
        "criteria": criteria,
        "note": (
            "485k corrects the 485j C1-only FAIL. FIX 1: the devalued-state OFC head "
            "driver now RE-RANKS (inverted high-threat outcome-coupling) instead of "
            "training flat (anti-range), so devaluation produces a differentiated "
            "devalued bias and ACTIVELY MOVES the committed selection. FIX 2: readiness "
            "now gates the DEVALUED-state bias range (the SAME statistic C1 routes on), "
            "so a below-floor devalued range self-routes substrate_not_ready_requeue, "
            "NEVER a false weakens. PRIMARY DVs measured on the COMMITTED selection "
            "through the MECH-448 demotion-enabled E3 selector "
            "(SelectionResult.selected_index), forced to the deterministic "
            "argmin-within-eligible path. Conversion is attributable to the demotion "
            "lever via the 3-arm dissociation: ARM_2 (trained head + demotion) must "
            "convert where ARM_1 (trained head, demotion off = F-dominance ceiling) and "
            "ARM_0 (frozen head + demotion = silence) do not. The demotion envelope "
            "(excluded_count, dn_sigma) is KEPT verbatim from 485j -- it fired correctly "
            "there (excluded_count=5); the C1 residual was the flat devalued head, NOT "
            "the envelope. Below the high-threat OR devalued bias-range floor OR with "
            "excluded_count == 0 -> substrate_not_ready_requeue (non_contributory), "
            "NEVER a false SD-033b/MECH-263 weakens. Readiness-met + DVs-fail is an "
            "honest result (conversion_ceiling_persists_despite_demotion -> "
            "/failure-autopsy / MECH-449 follow-on before stamping). Secondary readout: "
            "the high-vs-devalued OFC bias-VECTOR change (devaluation_bias_l2_shift / "
            "_cosine) records the valuation re-ranking even if the committed-argmax TV "
            "shift is thin."
        ),
    }


def _evidence_direction(result: Dict) -> str:
    if not result["readiness_met"]:
        return "non_contributory"          # substrate_not_ready_requeue, NOT a weakens
    return "supports" if result["acceptance"]["pass"] else "weakens"


def write_manifest(result: Dict, dry_run: bool, elapsed: float) -> Tuple[Path, str]:
    acc = result["acceptance"]
    interpretation = _interpretation(result)
    direction = _evidence_direction(result)
    outcome = "PASS" if (acc["pass"] and result["readiness_met"]) else "FAIL"
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.json"
    manifest = {
        "schema_version": "v1",
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "executing_hostname": socket.gethostname(),
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "result": outcome,
        "outcome": outcome,
        "evidence_direction": direction,
        "evidence_direction_per_claim": {
            "SD-033b": direction,
            "MECH-263": direction,
        },
        "non_degenerate": bool(result["readiness_met"]),
        "degeneracy_reason": (
            None if result["readiness_met"]
            else "substrate_not_ready_requeue: ARM_2 trained-head OFC bias range below "
                 "floor at the HIGH-THREAT (C2 positive control) OR DEVALUED (485k FIX 2, "
                 "the statistic C1 routes on) state, head untrained, or the CALIBRATED "
                 "MECH-448 envelope still admitted all candidates (excluded_count == 0 -- "
                 "flat/near-flat F, no clean share-gap to calibrate a floor) -> the "
                 "behavioural test never ran"
        ),
        "dry_run": dry_run,
        "elapsed_sec": elapsed,
        "acceptance": acc,
        "interpretation": interpretation,
        "readiness_met": result["readiness_met"],
        "ready_seeds": result["ready_seeds"],
        "ready_devalued_seeds": result["ready_devalued_seeds"],
        "nondegen_seeds": result["nondegen_seeds"],
        "n_seeds": result["n_seeds"],
        "max_test_head_delta": result["max_test_head_delta"],
        "max_test_bias_range": result["max_test_bias_range"],
        "max_test_bias_range_devalued": result["max_test_bias_range_devalued"],
        "max_test_deval_bias_l2_shift": result["max_test_deval_bias_l2_shift"],
        "min_test_deval_bias_cosine": result["min_test_deval_bias_cosine"],
        "max_test_excluded_count": result["max_test_excluded_count"],
        "max_test_between_context_tv": result["max_test_between_tv"],
        "max_test_deval_shift": result["max_test_deval_shift"],
        "max_ceiling_deval_shift": result["max_ceiling_deval_shift"],
        "max_ceiling_between_tv": result["max_ceiling_between_tv"],
        "arm_results": [r for arm in result["by_arm"].values() for r in arm],
        "substrate_under_test": (
            "SD-033b train_state_bias_head + ofc_bias_scale=0.5 + SD-056 "
            "e2_action_contrastive P0 + paired threat-conditioned driver with a 485k "
            "RE-RANKING devalued-state term (inverted outcome-coupling -- REPLACES the "
            "485h/485j anti-range variance penalty that trained the head flat-at-devalued), "
            "ROUTED through the MECH-448 (ARC-107) rank-preserving F->eligibility "
            "demotion-enabled E3 selector (use_f_eligibility_demotion / "
            "f_eligibility_dn_sigma=0.0), with f_eligibility_envelope_floor CALIBRATED "
            "per (arm, seed) to the behavioural bank's measured merit-share distribution "
            "(default 0.30 fallback; keep F-best top-k in [2,4]) -- the demotion lever + "
            "envelope calibration are KEPT VERBATIM from 485j (they fired correctly there, "
            "excluded_count=5)"
        ),
        "unblocks": (
            "commitment_closure:GAP-8 (SD-033b candidate -> provisional behavioural "
            "evidence -- the devaluation half, completing the 485j discrimination "
            "conversion); contributes to behavioral_diversity_isolation:GAP-I"
        ),
        "supersedes": "V3-EXQ-485j",
        "predecessor": (
            "V3-EXQ-485j (run_id v3_exq_485j_sd033b_demotion_envelope_calibrated_"
            "behavioural_20260621T180033Z_v3; RAN FAIL 2026-06-21 as a SPLIT result, "
            "failure_autopsy_V3-EXQ-485j_2026-06-21). C2 task-role discrimination "
            "CONVERTED (between_context_tv 1.0 on 2/3 seeds; ARM_1 demotion-off control "
            "= 0.0) -- first MECH-448 cross-substrate generalisation to the OFC channel. "
            "C1 devaluation FAILED: the OFC trained head's bias range collapsed to ~0.02 "
            "(< 0.05 floor) AT THE DEVALUED STATE because the paired driver's "
            "DEVALUED_ANTIRANGE_WEIGHT=0.5 trained it FLAT-at-devalued, so the "
            "correctly-firing demotion selector (excluded_count=5) had nothing to "
            "re-rank -> shift=0. Orthogonal to the envelope (which fired correctly)"
        ),
        "predecessors": [
            "V3-EXQ-485b (devaluation, representation-level)",
            "V3-EXQ-485c (task-role discrimination, representation-level)",
            "V3-EXQ-485d (trained-OFC-head substrate readiness)",
            "V3-EXQ-485e (trained-OFC-head behavioural; non_contributory range-starved)",
            "V3-EXQ-485f (trained-OFC-head behavioural; non_contributory readiness-gate miscalibration)",
            "V3-EXQ-485g (trained-OFC-head behavioural; substrate_ceiling, first non-vacuous test, zero conversion)",
            "V3-EXQ-485h (disambiguating redesign; FAIL/non_contributory, F-dominance conversion ceiling)",
            "V3-EXQ-485i (demotion-enabled; FAIL/substrate_not_ready_requeue, MECH-448 envelope all-admit no-op excluded_count==0)",
            "V3-EXQ-485j (demotion-envelope-calibrated; SPLIT -- C2 discrimination CONVERTED, C1 devaluation FAIL on the flat anti-range devalued head)",
        ],
        "notes": (
            "CORRECTED DEVALUATION arm. SUPERSEDES V3-EXQ-485j, a SPLIT result "
            "(failure_autopsy_V3-EXQ-485j_2026-06-21): C2 task-role discrimination "
            "CONVERTED (between_context_tv 1.0 on 2/3 seeds; ARM_1 demotion-off control "
            "= 0.0; ARM_0 silent) = the FIRST cross-substrate corroboration that the "
            "MECH-448 BG F->eligibility demotion lever generalises off GAP-A foraging to "
            "the OFC valuation channel; but C1 devaluation FAILED because the 485h/485j "
            "paired driver's DEVALUED_ANTIRANGE_WEIGHT=0.5 trained the OFC head FLAT at "
            "the devalued state (bias range ~0.02 < the 0.05 floor), so the "
            "correctly-firing demotion selector (excluded_count=5, winner!=F-argmin) had "
            "nothing to re-rank -> devaluation_selection_shift=0. The FAIL was "
            "C1-ONLY and ORTHOGONAL to the envelope (which fired correctly) -- so 485k "
            "does NOT recalibrate the demotion envelope. TWO 485k fixes: (1) the "
            "devalued-state OFC head DRIVER now RE-RANKS (inverted high-threat "
            "outcome-coupling -> favour high predicted harm at the devalued state via "
            "the SAME REINFORCE-over-candidates form), so devaluation produces a "
            "DIFFERENTIATED devalued bias that down-biases the previously-preferred "
            "candidate / up-biases an alternative (Rudebeck & Murray 2014; Dickinson & "
            "Balleine) and ACTIVELY MOVES the committed selection; (2) the C1 readiness "
            "precondition is RE-TARGETED from the high-threat range to the DEVALUED-state "
            "range (the SAME statistic C1 routes on), so a below-floor devalued range "
            "self-routes substrate_not_ready_requeue (non_contributory), NEVER a false "
            "weakens (the V3-EXQ-642 same-statistic miss). KEPT verbatim from 485j: "
            "ofc_bias_scale=0.5, SD-056 P0 warmup, the MECH-448 demotion-on config "
            "(dn_sigma=0.0) + the per-(arm,seed) f_eligibility_envelope_floor calibration "
            "with the excluded_count>0 HARD non-degeneracy gate, and the 3-arm "
            "dissociation. The PRIMARY C1 devaluation_selection_shift + C2 "
            "between-context TV are measured on the COMMITTED selection through the real "
            "E3.select() (SelectionResult.selected_index). Readiness gates: OFC bias "
            "range > 0.05 at BOTH the high-threat (C2 control) AND devalued (C1) states + "
            "head trained + MECH-448 non-degeneracy (excluded_count > 0 on >=2/3 seeds) "
            "self-route substrate_not_ready_requeue below floor, NEVER a false SD-033b/"
            "MECH-263 weakens. Secondary readout: high-vs-devalued OFC bias-VECTOR change "
            "(devaluation_bias_l2_shift / _cosine). PROMOTES NOTHING by itself; a PASS "
            "contributes provisional behavioural evidence (the devaluation half) and "
            "helps close behavioral_diversity_isolation:GAP-I."
        ),
    }
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=False,
        config=manifest.get("config"),
        seeds=None,
        script_path=Path(__file__),
    )
    return out_path, outcome


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    args = parser.parse_args()
    t0 = time.time()
    _preflight()
    seeds = args.seeds if args.seeds is not None else ([0] if args.dry_run else [0, 1, 2])
    result = run(seeds=seeds, dry_run=args.dry_run)
    elapsed = time.time() - t0
    out_path, outcome = write_manifest(result, args.dry_run, elapsed)
    acc = result["acceptance"]
    interp = _interpretation(result)
    print("\n=== V3-EXQ-485k SUMMARY ===", flush=True)
    print(f"  readiness_met={result['readiness_met']} (ready_seeds={result['ready_seeds']}/{result['n_seeds']}, ready_devalued_seeds={result['ready_devalued_seeds']}/{result['n_seeds']}, nondegen_seeds={result['nondegen_seeds']}/{result['n_seeds']})", flush=True)
    print(f"  max_test_bias_range_high_threat={result['max_test_bias_range']:.6f} (floor {BIAS_RANGE_FLOOR})", flush=True)
    print(f"  max_test_bias_range_devalued={result['max_test_bias_range_devalued']:.6f} (floor {BIAS_RANGE_FLOOR}; 485k FIX 2 -- the C1-routed statistic)", flush=True)
    print(f"  max_test_deval_bias_l2_shift={result['max_test_deval_bias_l2_shift']:.4f}  min_test_deval_bias_cosine={result['min_test_deval_bias_cosine']:.4f}", flush=True)
    print(f"  max_test_excluded_count={result['max_test_excluded_count']}", flush=True)
    _cal = [r.get("f_eligibility_envelope_floor_calibrated") for r in result["by_arm"][ARM_TEST]]
    _calok = [r.get("f_eligibility_calibration_succeeded") for r in result["by_arm"][ARM_TEST]]
    print(f"  ARM_2 calibrated_envelope_floors={_cal} succeeded={_calok}", flush=True)
    print(f"  ARM_2 deval_shift={result['max_test_deval_shift']:.4f} vs ARM_1 ceiling={result['max_ceiling_deval_shift']:.4f}", flush=True)
    print(f"  ARM_2 between_tv={result['max_test_between_tv']:.4f} vs ARM_1 ceiling={result['max_ceiling_between_tv']:.4f}", flush=True)
    print(f"  C1_devaluation_behavioural_shift={acc['C1_devaluation_behavioural_shift']} ({acc['n_c1_seeds']}/{result['n_seeds']})", flush=True)
    print(f"  C2_discrimination_behavioural_separation={acc['C2_discrimination_behavioural_separation']} ({acc['n_c2_seeds']}/{result['n_seeds']})", flush=True)
    print(f"  C3_silence_control={acc['C3_silence_control']} ({acc['n_c3_seeds']}/{result['n_seeds']})", flush=True)
    print(f"  interpretation: {interp['label']}", flush=True)
    print(f"  evidence_direction: {_evidence_direction(result)}", flush=True)
    print(f"  outcome: {outcome}", flush=True)
    print(f"Result written to: {out_path}", flush=True)
    return out_path, outcome


if __name__ == "__main__":
    out_path, outcome = main()
    _raw = str(outcome).upper()
    emit_outcome(
        outcome=_raw if _raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=str(out_path),
        dry_run="--dry-run" in sys.argv,
    )
