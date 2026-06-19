#!/opt/local/bin/python3
"""V3-EXQ-485h: SD-033b trained-OFC-head BEHAVIOURAL validation (MECH-263 a+b) --
DISAMBIGUATING REDESIGN.

commitment_closure:GAP-8 evidence-grade follow-on. SUPERSEDES V3-EXQ-485g, the
FIRST non-vacuous test of the series, which was adjudicated
substrate_ceiling / non_contributory (NOT a weakens) by the confirmed
failure_autopsy_V3-EXQ-485g_2026-06-19.

WHY 485g did not close GAP-8 (verified from its manifest + autopsy, NOT a
falsification):
  485g readiness was MET (trained-head OFC cross-candidate bias RANGE 0.171 >=
  0.05 floor on 2/3 seeds; head weight-delta 6.32 genuinely trained) and the C3
  frozen-head silence control PASSed -- but C1 devaluation_selection_shift
  {0.001, 0.0, 0.010} << 0.05 (0/3) and C2 between_context_tv ~0 (degenerate,
  0/3). A GENUINE 0.17 cross-candidate bias RANGE with ZERO behavioural
  conversion. The autopsy left TWO undisambiguated readings the 485-series has
  never been able to separate:
    (T) THREAT-CONDITIONING GAP (lead reading): the 485g outcome-coupled driver
        adv_k = gain * threat * (mean_harm - harm_k) is ZERO at threat ~ 0, so
        the head got NO gradient to flatten its bias range when the outcome is
        devalued -- the exact suppression C1 measures. The head may have learned
        a threat-INVARIANT bias (range stays ~0.17 at devaluation), so devaluing
        z_harm barely changes compute_bias -> devaluation_selection_shift ~ 0.
    (F) F-DOMINANCE CONVERSION CEILING (MECH-439): plausible by analogy to the
        broader conversion-ceiling pattern, but NOT directly measurable in 485g
        -- the readout is the ISOLATED OFC softmax (modulatory-bias-selection-
        authority OFF), so the E3 primary score F is not in the loop.
  The decisive disambiguator -- the OFC bias RANGE AT the devalued state (and
  per task-role context) -- was never recorded. 485g reports bias_range_high
  (0.17) but not the devalued-state range, so T vs F is open.

THE THREE 485h ADDITIONS (failure_autopsy_V3-EXQ-485g_2026-06-19 section 6):

  (1) RECORD the disambiguator. _behavioural_eval now records the trained-head
      OFC bias cross-candidate RANGE AT the DEVALUED state (z_harm -> 0,
      bias_range_devalued) and PER task-role context (bias_range_context_a /
      _context_b), alongside the existing bias_range_high. Reading:
        - devalued range stays ~high (devaluation_range_collapse_ratio high)
          AND C1 fails  -> (T) the head learned a threat-INVARIANT bias.
        - devalued range collapses (~0) but selection still does not shift in the
          LIVE-E3 loop -> (F) conversion ceiling (tested by the optional arm (3),
          NOT by this isolated-softmax readout where a range collapse trivially
          flattens sel_low).

  (2) CLOSE THE DRIVER GAP (non-circular). The 485g spread-only driver is
      replaced by a PAIRED threat-conditioned driver
      (_ofc_threat_conditioned_driver_loss) that trains the head on BOTH regimes
      the eval tests, using ONE consistent construction:
        for each sampled snapshot (real candidate bank + captured z_world_rep):
          # high-threat regime (the spread target -- the 485g outcome-coupled term)
          settle state_code with z_world_rep + canonical z_harm_HIGH
          harm_k   = E3.harm_eval(bank_k)                  # [K] trained per-cand harm
          adv_k    = gain * |z_harm_high| * (mean(harm_k) - harm_k)   # low-harm favoured
          bias_hi  = ofc.compute_bias(bank)
          L_spread += -(adv_k * log_softmax(-bias_hi)).sum()         # range UNDER threat
          # devalued regime (the anti-range target -- the 485g-missing gradient)
          settle state_code onward with z_harm_LOW (= 0)
          bias_dev = ofc.compute_bias(bank)
          L_anti  += variance(bias_dev)                              # range -> small when devalued
        loss = mean(L_spread) + DEVALUED_ANTIRANGE_WEIGHT * mean(L_anti)
      NON-CIRCULARITY: the anti-range term penalises the bias-range MAGNITUDE at
      the devalued state_code; it does NOT inject any specific sel_low
      distribution or a preferred candidate. It encodes the genuine devaluation
      hypothesis -- no aversive outcome -> no differential harm-avoidance bias --
      grounded in the outcome structure. C1 is NOT trivially equal to readiness:
      the head must learn to produce a large range at the HIGH-z_harm state_code
      AND a small range at the ZERO-z_harm state_code using ONLY the state_code
      difference (the SD-033b z_harm-in-state_code mechanism, ofc_harm_dim>0). If
      the state_code does not carry enough z_harm signal, or the head cannot
      condition on it, the contrast fails honestly. A passing C1 is therefore a
      genuine threat-conditioned valuation, and a failing C1 with readiness MET
      is an honest result (governance + /failure-autopsy adjudicate T vs F using
      the recorded devalued-state range + the live-E3 arm).
      KEY DIFFERENCE vs 485g: the head now SEES a canonical high-threat AND a
      devalued state_code during training (485g only saw the moderate-threat
      rollout snapshots and never a devalued state_code, leaving its devalued
      output untrained / extrapolated -- the autopsy root cause).

  (3) OPTIONAL LIVE-E3 ARM (supplementary, diagnostic-only). For the trained
      (ARM_1) agent, _live_e3_devaluation_readout routes the OFC bias through the
      REAL E3 selector (agent.e3.select with use_modulatory_selection_authority
      ON, OFC bias passed as score_bias so it is rescaled into the F-dominated
      primary scores) and measures whether the COMMITTED selection shifts under
      devaluation, vs an F-only (no-OFC-bias) control. NON-VACUITY PRECONDITION:
      the readout only carries a verdict when modulatory_authority_active is True
      with a non-zero authority range at the high-threat state (the OFC bias
      genuinely reached the accumulator). If the OFC bias reaches the accumulator
      but does NOT move committed selection (vs F-only AND high-vs-devalued), that
      is the positive-adjacent test of the (F) F-dominance reading -> 485g/485h
      join the MECH-439 conversion-ceiling cluster. This arm NEVER gates the
      SD-033b/MECH-263 verdict and NEVER drives a weakens; it records the F-test
      data the autopsy deferred the F-vs-T call to.

KEPT FROM 485g: ofc_bias_scale=0.5, SD-056 e2_action_contrastive armed in P0,
  the absolute between_context_tv floor, the readiness floor 0.05 (==
  DEVAL_SHIFT_MARGIN, the same-statistic non-vacuity gate), the 2-arm
  ARM_0_frozen_head / ARM_1_trainable_head x 3-seed design, the OFC-isolated
  SD-054 bipartite reef/forage env (ofc_harm_dim>0), phased training
  (P0 encoder warmup WITH SD-056 contrastive -> P1 head training on the
  frozen-encoder state_code -> P2 BEHAVIOURAL eval), and LR_OFC_BIAS 2e-3 /
  P1 120 ep.

PRE-REGISTERED ACCEPTANCE (per seed, then >= MIN_PASS_SEEDS of 3):
  READINESS (non-vacuity precondition, SAME statistic the load-bearing criteria
    route on): trained-head bias cross-candidate RANGE at the high-threat positive
    control > BIAS_RANGE_FLOOR (0.05) AND head trained (weight-delta > floor).
    Below floor -> substrate_not_ready_requeue, NEVER a SD-033b/MECH-263 weakens.
  C1_devaluation_behavioural_shift (load-bearing): ARM_1 devaluation_selection_shift
    > DEVAL_SHIFT_MARGIN AND > ARM_0 (frozen) shift + DEVAL_SHIFT_MARGIN.
  C2_discrimination_behavioural_separation (load-bearing): ARM_1 separation_ratio
    >= SEPARATION_RATIO_MIN AND between_context_selection_tv >= BETWEEN_TV_FLOOR.
  C3_frozen_head_silent (load-bearing control): ARM_0 shift < 1e-9 AND ARM_0
    between-context separation < 1e-9.
  PASS (supports SD-033b + MECH-263) = READINESS AND C1 AND C2 AND C3 on a majority.

INTERPRETATION GRID (one row per plausible outcome -> next action):
  PASS / supports: trained head produces devaluation-sensitive AND task-role-
    discriminative candidate selection that the frozen head does not. SD-033b
    candidate -> provisional behavioural evidence; closes commitment_closure:GAP-8.
  FAIL / weakens (readiness MET, DVs fail): the trained head has genuine
    cross-candidate range (>= 0.05) but does NOT steer selection in a devaluation /
    discrimination way despite the driver fix. Route /failure-autopsy with the
    recorded devalued-state range + live-E3 F-test to adjudicate (T) threat-
    invariant-bias vs (F) F-dominance conversion ceiling BEFORE stamping. An
    honest result, not a vacuity artifact.
  FAIL / substrate_not_ready_requeue (readiness UNMET): the trained head still
    cannot drive bias range to the 0.05 DV floor -> the behavioural test never ran.
    Re-queue with a stronger driver / larger budget. NOT a weakens.

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

EXPERIMENT_TYPE = "v3_exq_485h_sd033b_trained_ofc_head_behavioural"
EXPERIMENT_PURPOSE = "evidence"
QUEUE_ID = "V3-EXQ-485h"
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
P1_EPISODES = 120                      # kept from 485g: give the driver budget
STEPS_PER_EPISODE = 100
TRAIN_EPS = P0_EPISODES + P1_EPISODES  # progress denominator M

LR_E1 = 1e-3
LR_E2_WF = 1e-3
LR_E3_HARM = 1e-3
LR_OFC_BIAS = 2e-3                     # kept from 485g
BATCH_SIZE = 32
WF_BUF_MAX = 256
N_PROBE_CANDIDATES = 8
RECORD_EVERY_N_STEPS = 4
OUTCOME_BUF_MAX = 512
POLICY_TEMPERATURE = 1.0
EMA_DECAY = 0.9

# -- 485h Fix (2): paired threat-conditioned driver --
OUTCOME_COUPLE_GAIN = 4.0   # gain on the per-candidate threat-weighted advantage
ADV_MIN_THRESHOLD = 1e-4    # min per-candidate harm spread for the spread term to fire
DEVALUED_ANTIRANGE_WEIGHT = 0.5   # weight on the devalued-state anti-range (variance) term
DRIVER_SNAP_BATCH = 16      # snapshots per P1 driver step (each runs 2 settle passes)

# -- 485f Fix 1 (kept): defeat OFC clamp-saturation (485e railed at +/-0.1) --
OFC_BIAS_SCALE = 0.5

# -- 485f Fix 2 (kept): SD-056 e2_action_contrastive in P0 -> action-divergent bank --
E2_CONTRASTIVE_WEIGHT = 0.05
E2_CONTRASTIVE_TEMP = 0.1

# -- P2 behavioural-eval drive lengths (parallel to 485b/485c/485f/485g) --
PRE_ONSET_TICKS = 25
POST_ONSET_TICKS = 40
CONTEXT_TICKS = 25

# -- Pre-registered thresholds (NOT derived from the run's own statistics) --
HEAD_DELTA_MIN = 1e-3
BIAS_RANGE_FLOOR = 0.05      # == DEVAL_SHIFT_MARGIN (same-statistic non-vacuity gate)
DEVAL_SHIFT_MARGIN = 0.05
SEPARATION_RATIO_MIN = 3.0
BETWEEN_TV_FLOOR = 0.05
FROZEN_SILENCE_EPS = 1e-9
MIN_PASS_SEEDS = 2

# -- 485h Fix (1): devalued-range collapse disambiguation thresholds --
DEVAL_RANGE_COLLAPSE_RATIO = 0.3   # devalued/high range below this -> range collapsed (-> F test)
THREAT_INVARIANT_RATIO = 0.6       # devalued/high range above this -> threat-invariant bias (-> T)

# -- 485h Fix (3): live-E3 supplementary F-test (diagnostic-only) --
LIVE_E3_AUTHORITY_RANGE_FLOOR = 1e-4   # OFC bias must reach the accumulator with this range
LIVE_E3_COMMITTED_SHIFT_EPS = 1e-3     # committed-selection TV below this == "did not move"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _action_to_onehot(idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
    return v


def _tv(p: torch.Tensor, q: torch.Tensor) -> float:
    """Total-variation distance between two selection distributions."""
    return float(0.5 * (p - q).abs().sum().item())


def _selection_dist(agent: REEAgent, bank: torch.Tensor, temperature: float) -> torch.Tensor:
    """softmax(-bias / T) over the candidate bank -- the OFC head's selection in
    isolation (REE lower-is-better -> lower bias = more favoured)."""
    bias = agent.ofc.compute_bias(bank)  # [K], clamped to +/-ofc_bias_scale
    return F.softmax(-bias / temperature, dim=0).detach()


def _bias_range(agent: REEAgent, bank: torch.Tensor) -> float:
    """Cross-candidate range (max - min) of the OFC bias over the bank -- the
    statistic the load-bearing selection DVs route on."""
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


def _settle_state_code(
    agent: REEAgent,
    z_world_rep: torch.Tensor,
    z_harm: Optional[torch.Tensor],
    ticks: int,
    reset: bool,
) -> None:
    """Drive the OFC state_code toward a regime via gated EMA updates.

    SHARED by the P1 driver (485h Fix 2) and the P2 eval (485h Fix 1) so the
    trained and tested state_codes are constructed identically. reset=True zeros
    the state_code first (entry to the high-threat regime); reset=False continues
    from the current state_code (the high -> devalued transition, mirroring the
    eval's PRE_ONSET-then-POST_ONSET trajectory)."""
    if reset:
        agent.ofc.reset()
    for _ in range(ticks):
        agent.ofc.update(z_world=z_world_rep, z_harm=z_harm, gate=1.0)


def _make_agent(seed: int, train_head: bool) -> Tuple[REEAgent, CausalGridWorldV2]:
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
    )
    config.e3.commitment_threshold = 0.5
    config.heartbeat.beta_gate_bistable = True
    agent = REEAgent(config)
    return agent, env


def _preflight() -> None:
    frozen, _ = _make_agent(1, train_head=False)
    trainable, _ = _make_agent(2, train_head=True)
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
    # 485h Fix (3) contract: E3Config carries the modulatory-authority flag so the
    # live-E3 readout can toggle it. Direct attr (read non-defensively in select()).
    assert hasattr(trainable.e3.config, "use_modulatory_selection_authority")
    del frozen, trainable
    print(
        "Preflight PASS: OFC analog + harm_dim + trainable head + bias_head_parameters "
        "+ ofc_bias_scale=0.5 + SD-056 contrastive + readiness floor 0.05 == DV floor "
        "+ paired threat-conditioned driver + modulatory-authority flag present",
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
            # SD-056 action-contrastive auxiliary loss (kept from 485f/485g).
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
    """485h Fix (2): PAIRED threat-conditioned, outcome-coupled, per-candidate
    driver. Trains the trained-OFC head on BOTH the high-threat regime (spread
    bias to favour low predicted harm -- the 485g outcome-coupled term) AND the
    devalued regime (suppress bias range -- the gradient 485g lacked), using a
    SHARED state_code construction (`_settle_state_code`) identical to the P2
    eval.

    snap_buf entries: (bank_snapshot [K, world_dim], z_world_rep [1, world_dim]).
    z_harm_high / z_harm_low: the canonical high-threat / devalued z_harm the eval
    drives the state_code with (so the trained regimes match the tested regimes).

    For each sampled snapshot:
      # high-threat regime -- spread (outcome-coupled REINFORCE-over-candidates)
      settle state_code (reset) with z_world_rep + z_harm_high   [PRE_ONSET_TICKS]
      harm_k = E3.harm_eval(bank_k)                              # [K] trained harm cost
      adv_k  = gain * |z_harm_high| * (mean(harm_k) - harm_k)    # low-harm favoured
      bias_hi = ofc.compute_bias(bank)                           # grad -> head
      L_spread += -(adv_k * log_softmax(-bias_hi)).sum()
      # devalued regime -- anti-range (variance penalty; non-circular)
      settle state_code (continue) with z_world_rep + z_harm_low [POST_ONSET_TICKS]
      bias_dev = ofc.compute_bias(bank)                          # grad -> head
      L_anti += variance(bias_dev)
    loss = mean(L_spread) + DEVALUED_ANTIRANGE_WEIGHT * mean(L_anti)

    The anti-range term penalises bias-range MAGNITUDE at the devalued state_code
    -- it injects NO specific sel_low distribution, only the genuine devaluation
    hypothesis (no aversive outcome -> indifference). The head must learn to read
    z_harm from the state_code to keep range high at z_harm_high and small at
    z_harm_low; that contrast can fail honestly. Returns (loss, n_spread, n_anti).

    Side effect: mutates ofc.state_code (settle passes); the caller saves/restores
    it so the rollout state is not corrupted."""
    if agent.ofc is None or len(snap_buf) < 2:
        return torch.zeros(1, device=device), 0, 0
    threat_high = float(z_harm_high.detach().norm().item())
    idxs = np.random.choice(
        len(snap_buf), size=min(DRIVER_SNAP_BATCH, len(snap_buf)), replace=False
    )
    spread_terms: List[torch.Tensor] = []
    anti_terms: List[torch.Tensor] = []
    for i in idxs:
        bank, z_world_rep = snap_buf[int(i)]
        if bank.shape[0] < 2:
            continue
        # ---- high-threat regime: spread ----
        _settle_state_code(agent, z_world_rep, z_harm_high, PRE_ONSET_TICKS, reset=True)
        with torch.no_grad():
            harm = agent.e3.harm_eval(bank).reshape(-1)  # [K] per-candidate harm cost
            if harm.numel() != bank.shape[0]:
                continue
            adv = OUTCOME_COUPLE_GAIN * threat_high * (harm.mean() - harm)  # [K]
        if float(adv.abs().max()) >= ADV_MIN_THRESHOLD:
            bias_hi = agent.ofc.compute_bias(bank)  # [K], grad flows into the head
            log_p = F.log_softmax(-bias_hi / POLICY_TEMPERATURE, dim=0)
            spread_terms.append(-(adv * log_p).sum())
        # ---- devalued regime: anti-range (continue the high->devalued transition) ----
        _settle_state_code(agent, z_world_rep, z_harm_low, POST_ONSET_TICKS, reset=False)
        bias_dev = agent.ofc.compute_bias(bank)  # [K], grad flows into the head
        anti_terms.append((bias_dev - bias_dev.mean()).pow(2).mean())
    if not spread_terms and not anti_terms:
        return torch.zeros(1, device=device), 0, 0
    loss = torch.zeros(1, device=device)
    if spread_terms:
        loss = loss + torch.stack(spread_terms).mean()
    if anti_terms:
        loss = loss + DEVALUED_ANTIRANGE_WEIGHT * torch.stack(anti_terms).mean()
    return loss, len(spread_terms), len(anti_terms)


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
    n_anti_terms = 0
    # snap_buf entries: (bank_snapshot, z_world_rep)
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
            # The paired driver settles the OFC state_code; save/restore so the
            # next episode's rollout starts clean (reset() also runs at ep top).
            saved_state = agent.ofc.state_code.detach().clone()
            l_loss, n_sp, n_an = _ofc_threat_conditioned_driver_loss(
                agent, snap_buf, z_harm_high, z_harm_low, device
            )
            with torch.no_grad():
                agent.ofc.state_code.copy_(saved_state)
            n_spread_terms += n_sp
            n_anti_terms += n_an
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
        "n_antirange_loss_terms": n_anti_terms,
    }


# --------------------------------------------------------------------------- #
# P2 behavioural eval
# --------------------------------------------------------------------------- #
def _collect_eval_candidates(
    agent: REEAgent, env: CausalGridWorldV2, steps: int
) -> Tuple[Optional[List], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Run the trained agent briefly and return (candidates[:K], candidate_bank
    [K, world_dim], z_world_rep [1, world_dim]) -- a real candidate list + its
    first-step z_world bank + a representative state for the controlled
    devaluation / discrimination drives. The candidates list is returned aligned
    with the bank so the live-E3 arm can pass an aligned score_bias."""
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


def _behavioural_eval(
    agent: REEAgent, bank: torch.Tensor, z_world_rep: torch.Tensor,
    z_harm_high: torch.Tensor, z_harm_low: torch.Tensor, seed: int,
) -> Dict:
    """On the trained agent: measure devaluation-selection shift + task-role
    discrimination separation from the OFC head's induced candidate selection.
    Also records the readiness positive control (bias range at the high-threat
    state) AND -- 485h Fix (1) -- the bias range AT the devalued state and per
    task-role context (the T-vs-F disambiguator the series never recorded)."""
    agent.eval()
    with torch.no_grad():
        # ---- P2-a devaluation sensitivity ----
        _settle_state_code(agent, z_world_rep, z_harm_high, PRE_ONSET_TICKS, reset=True)
        sel_high = _selection_dist(agent, bank, POLICY_TEMPERATURE)
        bias_range_high = _bias_range(agent, bank)  # readiness positive control
        _settle_state_code(agent, z_world_rep, z_harm_low, POST_ONSET_TICKS, reset=False)
        sel_low = _selection_dist(agent, bank, POLICY_TEMPERATURE)
        bias_range_devalued = _bias_range(agent, bank)  # 485h Fix (1): the disambiguator
        deval_shift = _tv(sel_high, sel_low)

        # ---- P2-b task-role discrimination ----
        ctx = CONTEXT_TICKS
        g = torch.Generator().manual_seed(20000 + seed)
        hist_a = z_world_rep + 1.5 + 0.1 * torch.randn(ctx, WORLD_DIM, generator=g)
        hist_b = z_world_rep - 1.5 + 0.1 * torch.randn(ctx, WORLD_DIM, generator=g)
        hist_a2 = z_world_rep + 1.5 + 0.1 * torch.randn(ctx, WORLD_DIM, generator=g)

        def _drive_then_select(hist: torch.Tensor) -> Tuple[torch.Tensor, float]:
            agent.ofc.reset()
            for t in range(hist.shape[0]):
                agent.ofc.update(z_world=hist[t:t + 1], z_harm=None, gate=1.0)
            agent.ofc.update(z_world=z_world_rep, z_harm=None, gate=1.0)
            sel = _selection_dist(agent, bank, POLICY_TEMPERATURE)
            rng = _bias_range(agent, bank)  # 485h Fix (1): per-context bias range
            return sel, rng

        sel_a, bias_range_context_a = _drive_then_select(hist_a)
        sel_b, bias_range_context_b = _drive_then_select(hist_b)
        sel_a2, _ = _drive_then_select(hist_a2)
        between = _tv(sel_a, sel_b)
        within = _tv(sel_a, sel_a2)
        separation_ratio = between / max(within, 1e-6)

    collapse_ratio = (
        bias_range_devalued / bias_range_high if bias_range_high > 1e-9 else 0.0
    )
    return {
        "devaluation_selection_shift": deval_shift,
        "bias_range_high_threat": bias_range_high,
        "bias_range_devalued": bias_range_devalued,
        "devaluation_range_collapse_ratio": collapse_ratio,
        "bias_range_context_a": bias_range_context_a,
        "bias_range_context_b": bias_range_context_b,
        "bank_zworld_spread": _bank_zworld_spread(bank),
        "between_context_selection_tv": between,
        "within_context_selection_jitter": within,
        "discrimination_separation_ratio": separation_ratio,
    }


def _live_e3_devaluation_readout(
    agent: REEAgent, cands: List, bank: torch.Tensor, z_world_rep: torch.Tensor,
    z_harm_high: torch.Tensor, z_harm_low: torch.Tensor,
) -> Dict:
    """485h Fix (3): OPTIONAL live-E3 supplementary F-test (diagnostic-only).

    Routes the OFC bias through the REAL E3 selector with modulatory-bias-
    selection-authority ON (OFC bias passed as score_bias so the authority
    rescales it into the F-dominated primary scores), and measures whether the
    COMMITTED selection (softmax over the post-authority scores) shifts under
    devaluation, vs an F-only (no-OFC-bias) control.

    NON-VACUITY PRECONDITION: a verdict is only carried when the authority is
    genuinely active with a non-zero modulatory range at the high-threat state
    (the OFC bias reached the accumulator). This arm NEVER gates the
    SD-033b/MECH-263 verdict and NEVER drives a weakens.

    Verdict:
      live_e3_not_ready: authority did not engage (no F-test possible).
      f_dominance_positive_adjacent: OFC bias reached the accumulator but moved
        neither the high-vs-F-only nor the high-vs-devalued committed selection
        -> the (F) F-dominance reading; 485g/485h join the MECH-439 cluster.
      live_e3_devaluation_shift_present: committed selection DID shift under
        devaluation through the live selector (would be SD-033b-positive in the
        live loop) -- the F-dominance reading is NOT supported by this run.
      live_e3_inconclusive: otherwise (OFC bias moved selection vs F-only but the
        devaluation contrast is ambiguous)."""
    out: Dict[str, object] = {
        "live_e3_status": "ok",
        "live_e3_label": "live_e3_not_ready",
        "live_e3_authority_active_high": False,
        "live_e3_authority_range_high": 0.0,
        "live_e3_committed_shift_devaluation": 0.0,
        "live_e3_committed_shift_vs_f_only": 0.0,
    }
    if cands is None or bank is None or z_world_rep is None or len(cands) < 2:
        out["live_e3_status"] = "no_eval_context"
        return out

    e3cfg = agent.e3.config
    prev_authority = bool(getattr(e3cfg, "use_modulatory_selection_authority", False))
    try:
        agent.eval()
        with torch.no_grad():
            # Authority ON so the bounded OFC bias is rescaled into the F scores.
            e3cfg.use_modulatory_selection_authority = True

            def _committed_dist(score_bias: Optional[torch.Tensor]) -> torch.Tensor:
                agent.e3.select(
                    cands,
                    temperature=POLICY_TEMPERATURE,
                    goal_state=getattr(agent, "goal_state", None),
                    score_bias=score_bias,
                )
                scores = agent.e3.last_scores.detach()
                return F.softmax(-scores / POLICY_TEMPERATURE, dim=0)

            # F-only control (no OFC bias).
            sel_f_only = _committed_dist(None)

            # High-threat: OFC bias routed through the live selector.
            _settle_state_code(agent, z_world_rep, z_harm_high, PRE_ONSET_TICKS, reset=True)
            ofc_bias_high = agent.ofc.compute_bias(bank).detach()
            sel_high_live = _committed_dist(ofc_bias_high)
            diag_high = dict(agent.e3.last_score_diagnostics)

            # Devalued: same trained head, devalued state_code.
            _settle_state_code(agent, z_world_rep, z_harm_low, POST_ONSET_TICKS, reset=False)
            ofc_bias_low = agent.ofc.compute_bias(bank).detach()
            sel_low_live = _committed_dist(ofc_bias_low)

        authority_active = bool(diag_high.get("modulatory_authority_active", False))
        authority_range = float(diag_high.get("modulatory_authority_range", 0.0))
        shift_deval = _tv(sel_high_live, sel_low_live)
        shift_vs_f = _tv(sel_high_live, sel_f_only)

        out["live_e3_authority_active_high"] = authority_active
        out["live_e3_authority_range_high"] = authority_range
        out["live_e3_committed_shift_devaluation"] = shift_deval
        out["live_e3_committed_shift_vs_f_only"] = shift_vs_f

        if not authority_active or authority_range < LIVE_E3_AUTHORITY_RANGE_FLOOR:
            out["live_e3_label"] = "live_e3_not_ready"
        elif (
            shift_vs_f < LIVE_E3_COMMITTED_SHIFT_EPS
            and shift_deval < LIVE_E3_COMMITTED_SHIFT_EPS
        ):
            out["live_e3_label"] = "f_dominance_positive_adjacent"
        elif shift_deval >= DEVAL_SHIFT_MARGIN:
            out["live_e3_label"] = "live_e3_devaluation_shift_present"
        else:
            out["live_e3_label"] = "live_e3_inconclusive"
    except Exception as exc:  # diagnostic-only arm: never break the primary verdict
        out["live_e3_status"] = f"error:{type(exc).__name__}"
        out["live_e3_label"] = "live_e3_not_ready"
    finally:
        e3cfg.use_modulatory_selection_authority = prev_authority
    return out


def run_arm_seed(arm: str, train_head: bool, seed: int, dry_run: bool) -> Dict:
    p0 = 3 if dry_run else P0_EPISODES
    p1 = 4 if dry_run else P1_EPISODES
    steps = 30 if dry_run else STEPS_PER_EPISODE
    print(f"\nSeed {seed} Condition {arm} ofc_train_state_bias_head={train_head}", flush=True)
    full_config = {
        "arm": arm,
        "train_head": train_head,
        "p0": p0,
        "p1": p1,
        "steps": steps,
        "ofc_bias_scale": OFC_BIAS_SCALE,
        "e2_contrastive_weight": E2_CONTRASTIVE_WEIGHT,
        "lr_ofc_bias": LR_OFC_BIAS,
        "outcome_couple_gain": OUTCOME_COUPLE_GAIN,
        "devalued_antirange_weight": DEVALUED_ANTIRANGE_WEIGHT,
        "bias_range_floor": BIAS_RANGE_FLOOR,
        "env_kwargs": ENV_KWARGS,
    }
    with arm_cell(seed, config_slice=full_config, script_path=Path(__file__)) as cell:
        agent, env = _make_agent(seed, train_head=train_head)
        # Canonical high-threat / devalued z_harm, built ONCE per (arm, seed) and
        # shared by the P1 driver and the P2 eval so trained regimes == tested.
        g = torch.Generator().manual_seed(20000 + seed)
        z_harm_high = torch.randn(1, HARM_DIM, generator=g) * 1.0
        z_harm_low = torch.zeros(1, HARM_DIM)
        for ep in range(p0):
            _encoder_step(agent, env, steps, arm, ep + 1)
        p1m = _p1_train(agent, env, p1, steps, arm, train_head, z_harm_high, z_harm_low)
        cands, bank, z_world_rep = _collect_eval_candidates(agent, env, steps)
        if bank is None or z_world_rep is None:
            beh = {
                "devaluation_selection_shift": 0.0,
                "bias_range_high_threat": 0.0,
                "bias_range_devalued": 0.0,
                "devaluation_range_collapse_ratio": 0.0,
                "bias_range_context_a": 0.0,
                "bias_range_context_b": 0.0,
                "bank_zworld_spread": 0.0,
                "between_context_selection_tv": 0.0,
                "within_context_selection_jitter": 0.0,
                "discrimination_separation_ratio": 0.0,
                "eval_context_built": False,
            }
            live = {"live_e3_status": "no_eval_context", "live_e3_label": "live_e3_not_ready"}
        else:
            beh = _behavioural_eval(agent, bank, z_world_rep, z_harm_high, z_harm_low, seed)
            beh["eval_context_built"] = True
            # 485h Fix (3): live-E3 F-test only on the trained head (ARM_1).
            if train_head:
                live = _live_e3_devaluation_readout(
                    agent, cands, bank, z_world_rep, z_harm_high, z_harm_low
                )
            else:
                live = {"live_e3_status": "frozen_arm_skipped", "live_e3_label": "n/a"}
        row = {
            "arm": arm,
            "seed": seed,
            "ofc_train_state_bias_head": train_head,
            "head_weight_delta_norm": p1m["head_weight_delta_norm"],
            "n_grad_nonzero_updates": p1m["n_grad_nonzero_updates"],
            "n_spread_loss_terms": p1m["n_spread_loss_terms"],
            "n_antirange_loss_terms": p1m["n_antirange_loss_terms"],
            "p1_mean_abs_ofc_bias": p1m["p1_mean_abs_ofc_bias"],
            **beh,
            **live,
        }
        cell.stamp(row)
    print(
        f"verdict: {'PASS' if (train_head and beh.get('eval_context_built')) else 'FAIL'}  "
        f"head_delta={p1m['head_weight_delta_norm']:.6f}  "
        f"bias_range_hi={beh['bias_range_high_threat']:.6f}  "
        f"bias_range_dev={beh['bias_range_devalued']:.6f}  "
        f"collapse_ratio={beh['devaluation_range_collapse_ratio']:.3f}  "
        f"deval_shift={beh['devaluation_selection_shift']:.4f}  "
        f"between_tv={beh['between_context_selection_tv']:.4f}  "
        f"live_e3={row.get('live_e3_label')}",
        flush=True,
    )
    return row


# --------------------------------------------------------------------------- #
# aggregation + interpretation
# --------------------------------------------------------------------------- #
def run(seeds: Optional[List[int]] = None, dry_run: bool = False) -> Dict:
    if seeds is None:
        seeds = [0, 1, 2]
    arms = [
        ("ARM_0_frozen_head", False),
        ("ARM_1_trainable_head", True),
    ]
    by_arm: Dict[str, List[Dict]] = {a[0]: [] for a in arms}
    for seed in seeds:
        for arm_label, train_head in arms:
            by_arm[arm_label].append(run_arm_seed(arm_label, train_head, seed, dry_run))

    frozen = by_arm["ARM_0_frozen_head"]
    trained = by_arm["ARM_1_trainable_head"]
    n = len(seeds)

    ready_seeds = sum(
        1
        for r in trained
        if r["head_weight_delta_norm"] > HEAD_DELTA_MIN
        and r["bias_range_high_threat"] > BIAS_RANGE_FLOOR
        and r.get("eval_context_built", False)
    )
    readiness_met = ready_seeds >= MIN_PASS_SEEDS

    c1_seeds = 0
    c2_seeds = 0
    c3_seeds = 0
    for fr, tr in zip(frozen, trained):
        if (
            tr["devaluation_selection_shift"] > DEVAL_SHIFT_MARGIN
            and tr["devaluation_selection_shift"]
            > fr["devaluation_selection_shift"] + DEVAL_SHIFT_MARGIN
        ):
            c1_seeds += 1
        if (
            tr["discrimination_separation_ratio"] >= SEPARATION_RATIO_MIN
            and tr["between_context_selection_tv"] >= BETWEEN_TV_FLOOR
        ):
            c2_seeds += 1
        if (
            fr["devaluation_selection_shift"] < FROZEN_SILENCE_EPS
            and fr["between_context_selection_tv"] < FROZEN_SILENCE_EPS
        ):
            c3_seeds += 1

    c1 = c1_seeds >= MIN_PASS_SEEDS
    c2 = c2_seeds >= MIN_PASS_SEEDS
    c3 = c3_seeds >= MIN_PASS_SEEDS
    overall_pass = readiness_met and c1 and c2 and c3

    # 485h Fix (1): T-vs-F disambiguation summary over the trained arm.
    collapse_ratios = [
        r.get("devaluation_range_collapse_ratio", 0.0)
        for r in trained
        if r.get("eval_context_built", False)
    ]
    max_collapse_ratio = max(collapse_ratios, default=0.0)
    min_collapse_ratio = min(collapse_ratios, default=0.0)
    # threat-invariant: devalued range stays high on a majority of seeds
    n_threat_invariant = sum(1 for x in collapse_ratios if x >= THREAT_INVARIANT_RATIO)
    n_range_collapsed = sum(1 for x in collapse_ratios if x <= DEVAL_RANGE_COLLAPSE_RATIO)
    # 485h Fix (3): live-E3 F-test summary (trained arm).
    live_labels = [r.get("live_e3_label", "n/a") for r in trained]
    n_f_dominance = sum(1 for x in live_labels if x == "f_dominance_positive_adjacent")
    n_live_ready = sum(
        1 for r in trained if r.get("live_e3_authority_active_high", False)
    )

    return {
        "by_arm": by_arm,
        "n_seeds": n,
        "ready_seeds": ready_seeds,
        "readiness_met": readiness_met,
        "max_trained_head_delta": max((r["head_weight_delta_norm"] for r in trained), default=0.0),
        "max_trained_bias_range": max((r["bias_range_high_threat"] for r in trained), default=0.0),
        "max_trained_bias_range_devalued": max((r["bias_range_devalued"] for r in trained), default=0.0),
        "max_trained_bank_spread": max((r.get("bank_zworld_spread", 0.0) for r in trained), default=0.0),
        "max_trained_between_tv": max((r["between_context_selection_tv"] for r in trained), default=0.0),
        "max_collapse_ratio": max_collapse_ratio,
        "min_collapse_ratio": min_collapse_ratio,
        "n_threat_invariant_seeds": n_threat_invariant,
        "n_range_collapsed_seeds": n_range_collapsed,
        "n_f_dominance_seeds": n_f_dominance,
        "n_live_ready_seeds": n_live_ready,
        "acceptance": {
            "C1_devaluation_behavioural_shift": c1,
            "C2_discrimination_behavioural_separation": c2,
            "C3_frozen_head_silent": c3,
            "n_c1_seeds": c1_seeds,
            "n_c2_seeds": c2_seeds,
            "n_c3_seeds": c3_seeds,
            "pass": overall_pass,
        },
    }


def _disambiguation(result: Dict) -> Dict:
    """485h: record the T-vs-F disambiguation (NOT an auto-verdict). When readiness
    is met and C1 fails, the recorded devalued-state range + live-E3 F-test let
    governance / a follow-up /failure-autopsy route to (T) threat-invariant bias
    vs (F) F-dominance conversion ceiling (MECH-439)."""
    acc = result["acceptance"]
    ready = bool(result["readiness_met"])
    c1_failed = not acc["C1_devaluation_behavioural_shift"]
    reading = "not_applicable"
    if ready and c1_failed:
        if result["n_f_dominance_seeds"] >= MIN_PASS_SEEDS:
            reading = "F_dominance_conversion_ceiling_supported_route_MECH-439"
        elif result["n_threat_invariant_seeds"] >= MIN_PASS_SEEDS:
            reading = "T_threat_invariant_bias"
        elif result["n_range_collapsed_seeds"] >= MIN_PASS_SEEDS:
            reading = "devalued_range_collapsed_isolated_loop_route_live_E3"
        else:
            reading = "mixed_needs_failure_autopsy"
    elif ready and acc["pass"]:
        reading = "conversion_supported"
    return {
        "reading": reading,
        "note": (
            "Disambiguation is RECORDED, not auto-stamped. Below the 0.05 bias-range "
            "floor -> substrate_not_ready_requeue (non_contributory), NEVER a weakens. "
            "Readiness-met + C1-fail is an honest result with the driver gap closed; "
            "route /failure-autopsy with bias_range_devalued + the live-E3 F-test to "
            "adjudicate T (threat-invariant bias) vs F (F-dominance conversion ceiling, "
            "MECH-439) before stamping a SD-033b weakens."
        ),
        "max_trained_bias_range_high": float(result["max_trained_bias_range"]),
        "max_trained_bias_range_devalued": float(result["max_trained_bias_range_devalued"]),
        "max_collapse_ratio": float(result["max_collapse_ratio"]),
        "min_collapse_ratio": float(result["min_collapse_ratio"]),
        "n_threat_invariant_seeds": int(result["n_threat_invariant_seeds"]),
        "n_range_collapsed_seeds": int(result["n_range_collapsed_seeds"]),
        "n_f_dominance_seeds": int(result["n_f_dominance_seeds"]),
        "n_live_ready_seeds": int(result["n_live_ready_seeds"]),
    }


def _interpretation(result: Dict) -> Dict:
    acc = result["acceptance"]
    ready = bool(result["readiness_met"])
    if not ready:
        label = "substrate_not_ready_requeue"
    elif acc["pass"]:
        label = "sd033b_behavioural_functional_signature_supported"
    else:
        label = "sd033b_behavioural_functional_signature_absent"

    preconditions = [
        {
            "name": "ofc_trained_head_bias_cross_candidate_range_supra_dv_floor",
            "description": (
                "ARM_1 trained-head OFC bias cross-candidate RANGE (max-min over the "
                "real candidate bank) at the high-threat positive-control state clears "
                "BIAS_RANGE_FLOOR (0.05 == DEVAL_SHIFT_MARGIN, the same statistic the "
                "load-bearing DVs route on). RANGE not magnitude. Below floor -> "
                "substrate_not_ready_requeue (stronger driver / budget), NEVER a weakens."
            ),
            "measured": float(result["max_trained_bias_range"]),
            "threshold": BIAS_RANGE_FLOOR,
            "control": "trained-arm bias over a real candidate bank at the high-threat state",
            "met": result["max_trained_bias_range"] > BIAS_RANGE_FLOOR,
        },
        {
            "name": "ofc_head_weight_delta_supra_floor",
            "description": (
                "ARM_1 state_bias_head weight-delta-from-init L2 norm clears "
                "HEAD_DELTA_MIN -- the head genuinely trained under the 485h paired "
                "threat-conditioned driver."
            ),
            "measured": float(result["max_trained_head_delta"]),
            "threshold": HEAD_DELTA_MIN,
            "control": "trainable-arm head trained on frozen-encoder state_code in P1",
            "met": result["max_trained_head_delta"] > HEAD_DELTA_MIN,
        },
    ]
    criteria_non_degenerate = {
        "C1": bool(acc["C1_devaluation_behavioural_shift"]) and ready,
        "C2": bool(acc["C2_discrimination_behavioural_separation"]) and ready,
        "C3": bool(acc["C3_frozen_head_silent"]),
    }
    criteria = [
        {"name": "C1_devaluation_behavioural_shift", "load_bearing": True, "passed": bool(acc["C1_devaluation_behavioural_shift"])},
        {"name": "C2_discrimination_behavioural_separation", "load_bearing": True, "passed": bool(acc["C2_discrimination_behavioural_separation"])},
        {"name": "C3_frozen_head_silent", "load_bearing": True, "passed": bool(acc["C3_frozen_head_silent"])},
    ]
    return {
        "label": label,
        "preconditions": preconditions,
        "criteria_non_degenerate": criteria_non_degenerate,
        "criteria": criteria,
        "disambiguation": _disambiguation(result),
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
        "dry_run": dry_run,
        "elapsed_sec": elapsed,
        "acceptance": acc,
        "interpretation": interpretation,
        "readiness_met": result["readiness_met"],
        "ready_seeds": result["ready_seeds"],
        "n_seeds": result["n_seeds"],
        "max_trained_head_delta": result["max_trained_head_delta"],
        "max_trained_bias_range": result["max_trained_bias_range"],
        "max_trained_bias_range_devalued": result["max_trained_bias_range_devalued"],
        "max_trained_bank_zworld_spread": result["max_trained_bank_spread"],
        "max_trained_between_context_tv": result["max_trained_between_tv"],
        "disambiguation_summary": {
            "max_collapse_ratio": result["max_collapse_ratio"],
            "min_collapse_ratio": result["min_collapse_ratio"],
            "n_threat_invariant_seeds": result["n_threat_invariant_seeds"],
            "n_range_collapsed_seeds": result["n_range_collapsed_seeds"],
            "n_f_dominance_seeds": result["n_f_dominance_seeds"],
            "n_live_ready_seeds": result["n_live_ready_seeds"],
        },
        "arm_results": [r for arm in result["by_arm"].values() for r in arm],
        "substrate_under_test": "SD-033b train_state_bias_head + ofc_bias_scale=0.5 + SD-056 e2_action_contrastive P0 + paired threat-conditioned driver (spread@high + anti-range@devalued)",
        "unblocks": "commitment_closure:GAP-8 (SD-033b candidate -> provisional behavioural evidence)",
        "supersedes": "V3-EXQ-485g",
        "predecessor": "V3-EXQ-485g (substrate_ceiling / non_contributory; first non-vacuous test, conversion absent -- supersede)",
        "predecessors": [
            "V3-EXQ-485b (devaluation, representation-level)",
            "V3-EXQ-485c (task-role discrimination, representation-level)",
            "V3-EXQ-485d (trained-OFC-head substrate readiness)",
            "V3-EXQ-485e (trained-OFC-head behavioural; non_contributory range-starved)",
            "V3-EXQ-485f (trained-OFC-head behavioural; non_contributory readiness-gate miscalibration)",
            "V3-EXQ-485g (trained-OFC-head behavioural; substrate_ceiling, first non-vacuous test, zero conversion)",
        ],
        "notes": (
            "Evidence-grade trained-OFC-head BEHAVIOURAL validation of MECH-263 "
            "signatures a (devaluation sensitivity) + b (task-role discrimination), "
            "the DISAMBIGUATING REDESIGN superseding the substrate_ceiling V3-EXQ-485g "
            "per failure_autopsy_V3-EXQ-485g_2026-06-19. 485g produced a genuine 0.17 "
            "cross-candidate bias RANGE with ZERO behavioural conversion (readiness MET, "
            "C1 0/3, C2 degenerate) and left T (threat-conditioning gap) vs F "
            "(F-dominance conversion ceiling) undisambiguated. 485h adds THREE things: "
            "(1) records the OFC bias RANGE at the devalued state (bias_range_devalued) "
            "and per task-role context -- the T-vs-F disambiguator the series never "
            "measured; (2) closes the driver gap with a PAIRED threat-conditioned driver "
            "(outcome-coupled spread@high-threat-state_code + anti-range variance "
            "penalty@devalued-state_code, both built via the SAME _settle_state_code the "
            "eval uses), so the head is trained on BOTH regimes (485g only saw moderate "
            "rollout snapshots and never a devalued state_code) -- non-circular because "
            "the anti-range term injects only indifference-at-devaluation, not a specific "
            "sel_low; (3) an OPTIONAL live-E3 supplementary F-test (modulatory-bias-"
            "selection-authority ON, OFC bias as score_bias through the real selector) "
            "with a hard non-vacuity precondition, diagnostic-only, NEVER gating the "
            "verdict -- if the OFC bias reaches the accumulator but does not move "
            "committed selection there, 485g/485h join the MECH-439 conversion-ceiling "
            "cluster. Same-statistic non-vacuity gate: below 0.05 bias range -> "
            "substrate_not_ready_requeue (non_contributory), NEVER a false "
            "SD-033b/MECH-263 weakens; readiness-met-but-DVs-fail is an honest result "
            "(governance + /failure-autopsy adjudicate T vs F with the recorded "
            "disambiguation data). OFC is the sole bias channel in the primary arms; "
            "frozen-zeroed head is the GAP-1-analogue silence control."
        ),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
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
    print("\n=== V3-EXQ-485h SUMMARY ===", flush=True)
    print(f"  readiness_met={result['readiness_met']} (ready_seeds={result['ready_seeds']}/{result['n_seeds']})", flush=True)
    print(f"  max_trained_bias_range={result['max_trained_bias_range']:.6f} (floor {BIAS_RANGE_FLOOR})", flush=True)
    print(f"  max_trained_bias_range_devalued={result['max_trained_bias_range_devalued']:.6f}", flush=True)
    print(f"  collapse_ratio min/max={result['min_collapse_ratio']:.3f}/{result['max_collapse_ratio']:.3f}", flush=True)
    print(f"  C1_devaluation_behavioural_shift={acc['C1_devaluation_behavioural_shift']} ({acc['n_c1_seeds']}/{result['n_seeds']})", flush=True)
    print(f"  C2_discrimination_behavioural_separation={acc['C2_discrimination_behavioural_separation']} ({acc['n_c2_seeds']}/{result['n_seeds']})", flush=True)
    print(f"  C3_frozen_head_silent={acc['C3_frozen_head_silent']} ({acc['n_c3_seeds']}/{result['n_seeds']})", flush=True)
    print(f"  T-vs-F reading: {interp['disambiguation']['reading']}", flush=True)
    print(f"  live_e3 f_dominance_seeds={result['n_f_dominance_seeds']} ready_seeds={result['n_live_ready_seeds']}", flush=True)
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
    )
