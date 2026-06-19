#!/opt/local/bin/python3
"""V3-EXQ-485g: SD-033b trained-OFC-head BEHAVIOURAL validation (MECH-263 a+b).

commitment_closure:GAP-8 evidence-grade follow-on. SUPERSEDES V3-EXQ-485f, which
was reclassified weakens -> non_contributory (measurement_test_design_defect:
readiness-gate miscalibration) by failure_autopsy_batch9_2026-06-12.

WHY 485f did not close GAP-8 (verified from its manifest, NOT a falsification):
  The three 485e fixes all took effect -- ofc_bias_scale 0.5 defeated clamp-
  saturation, SD-056 e2_action_contrastive lifted the candidate-bank z_world
  spread off zero (max_trained_bank_zworld_spread = 0.043, per-arm 0.017-0.043,
  squarely in the 569d operative band 0.02-0.07; V3-EXQ-617 multistep readiness
  PASS all gates), and the absolute between_context_tv floor caught the C2
  vacuity. BUT the readiness gate (bias_range > BIAS_RANGE_FLOOR=1e-3) cleared by
  a HAIR (max_trained_bias_range = 0.00898) -- ~50x BELOW the 0.05 the devaluation
  / between-context DVs actually require. So "ready" certified a vacuous test
  (devaluation_selection_shift 5e-5..1.8e-4, between_context_tv ~2e-4).

  CRITICAL DIAGNOSIS: the candidate-bank z_world spread is NOT the binding
  constraint. The bank delivered 0.043; the OFC-head READOUT compressed that
  0.043-distinguishable input down to a 0.009 cross-candidate bias range and
  never approached its raised ofc_bias_scale=0.5 clamp rail. The bottleneck has
  moved DOWNSTREAM of the bank, into the head's training signal. (Daw 2005 dual-
  system / Wilson-Schoenbaum 2014: the representation is now distinguishable;
  the readout is the wall.)

  485f used _ofc_reinforce_loss = SHARED-RETURN, SELECTED-INDEX-ONLY REINFORCE
  (advantage = ep_return - single scalar baseline, applied to log_p[sel_idx]
  only, sparsified by ADV_MIN_THRESHOLD) -- the exact form 598b documents as
  "collapsed in 543l." It gives the head NO per-candidate, outcome-grounded,
  threat-conditioned signal, so the head learns almost no cross-candidate range.
  Raising the clamp again is useless (the head is not railing); the driver itself
  is the wall.

TWO FIXES FOR 485g:
  (a) READINESS FLOOR RE-ALIGNED to the DV-significance floor. BIAS_RANGE_FLOOR
      raised 1e-3 -> 0.05 (== DEVAL_SHIFT_MARGIN). The same-statistic non-vacuity
      gate can now NO LONGER certify a vacuous DV: "ready" means the trained-head
      bias cross-candidate RANGE is competitive with the selection softmax. If the
      head cannot reach 0.05 range, the ONLY correct route is
      substrate_not_ready_requeue (NEVER a weakens) -- the driver / budget needs
      more, the claim is not tested.

  (b) STRONGER OUTCOME-COUPLED, THREAT-CONDITIONED, PER-CANDIDATE driver replacing
      485f's under-driving shared-return REINFORCE. _ofc_outcome_coupled_range_loss
      is the SD-033a/598b "REINFORCE-over-candidates" pattern, sharpened to the
      OFC's MECH-263 devaluation function:
        for each sampled training snapshot (real candidate bank + captured threat):
          harm_k   = E3.harm_eval(bank_k)            # [K] per-candidate predicted
                                                     #     harm cost (trained in P0)
          adv_k    = gain * threat * (mean(harm_k) - harm_k)   # [K] low-harm
                                                     #     favoured, scaled by the
                                                     #     CURRENT threat magnitude
          bias     = ofc.compute_bias(bank)          # [K], gradient flows
          loss    += -(adv_k * log_softmax(-bias)).sum()   # REINFORCE over ALL K
      Distinguishing properties vs 485f's driver:
        - PER-CANDIDATE: trains the head over ALL K candidates (not just the
          selected index), so it must SPREAD bias across the bank -- the direct
          range driver. Outcome-grounded (uses the trained harm_eval, not injected
          noise), so a 0.05+ range that clears readiness is a GENUINE outcome
          mapping, never an artefact.
        - THREAT-CONDITIONED: the advantage is zero when threat ~ 0, so the head
          gets gradient to spread bias ONLY under threat. This gives the head the
          OPPORTUNITY to learn threat-conditioned valuation (range under threat,
          flat when devalued) -- which is exactly the MECH-263 devaluation
          signature. Whether the trained head SUPPRESSES range at low threat
          (-> non-zero devaluation shift) is what C1 tests; the driver does NOT
          inject the shift (the low-threat gradient is zero, not anti-range), so
          a passing C1 is non-circular and a failing C1 with readiness MET is an
          honest weakens.
      Budget raised to give the driver room: LR_OFC_BIAS 5e-4 -> 2e-3,
      P1_EPISODES 60 -> 120, ADV_MIN_THRESHOLD relaxed (the per-candidate loss
      keys on harm-spread, not on a high-advantage episode).

KEPT FROM 485f: ofc_bias_scale=0.5 (clamp headroom is fine, not the limit),
  SD-056 e2_action_contrastive armed in P0 (the bank-spread substrate, confirmed
  live), the absolute between_context_tv floor (BETWEEN_TV_FLOOR=0.05), the three
  pre-registered acceptance criteria (C1 devaluation behavioural shift, C2
  between-context discrimination non-degenerate, C3 frozen-head silence control),
  the 2-arm ARM_0_frozen_head / ARM_1_trainable_head x 3-seed design, the OFF-
  isolated SD-054 bipartite reef/forage env (ofc_harm_dim>0 so z_harm enters the
  state_code -- the aversive-devaluation-capable shape), and phased training
  (P0 encoder warmup WITH SD-056 contrastive -> P1 head training on the frozen-
  encoder state_code -> P2 BEHAVIOURAL eval).

  P2-a DEVALUATION SENSITIVITY (MECH-263 a): stable HIGH-threat z_harm -> sel_high;
    DEVALUE (z_harm -> 0) -> sel_low; devaluation_selection_shift = TV(sel_high,
    sel_low). Frozen head (bias==0) keeps a uniform selection -> shift exactly 0.
  P2-b TASK-ROLE DISCRIMINATION (MECH-263 b): two perceptually-matched task-role
    histories converging on the same matched final input; between = TV(sel_A,
    sel_B), within = TV(sel_A, sel_A2), separation_ratio = between / max(within, eps).
    Frozen head -> between == within == 0.

PRE-REGISTERED ACCEPTANCE (per seed, then >= MIN_PASS_SEEDS of 3):
  READINESS (non-vacuity precondition, SAME statistic the load-bearing criteria
    route on): trained-head bias cross-candidate RANGE at the high-threat positive
    control > BIAS_RANGE_FLOOR (now 0.05) AND head trained (weight-delta > floor).
    Below floor -> substrate_not_ready_requeue, NEVER a substrate verdict on SD-033b.
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
  FAIL / weakens (readiness MET, DVs fail): the trained head's bias has genuine
    cross-candidate range (>= 0.05) but does NOT steer selection in a devaluation /
    discrimination way -> the OFC head does not carry the MECH-263 behavioural
    signature. Route /failure-autopsy before stamping; an honest weakens.
  FAIL / substrate_not_ready_requeue (readiness UNMET): the trained head still
    cannot drive bias range to the 0.05 DV floor (driver / budget too weak, or the
    bank spread regressed) -> the behavioural test never ran. Re-queue with a
    stronger driver / larger P1 budget / explicit threat-curriculum. NOT a weakens.

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

EXPERIMENT_TYPE = "v3_exq_485g_sd033b_trained_ofc_head_behavioural"
EXPERIMENT_PURPOSE = "evidence"
QUEUE_ID = "V3-EXQ-485g"
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
P1_EPISODES = 120                      # 485g Fix (b): 60 -> 120, give the driver budget
STEPS_PER_EPISODE = 100
TRAIN_EPS = P0_EPISODES + P1_EPISODES  # progress denominator M

LR_E1 = 1e-3
LR_E2_WF = 1e-3
LR_E3_HARM = 1e-3
LR_OFC_BIAS = 2e-3                     # 485g Fix (b): 5e-4 -> 2e-3, stronger head training
BATCH_SIZE = 32
WF_BUF_MAX = 256
N_PROBE_CANDIDATES = 8
RECORD_EVERY_N_STEPS = 4
OUTCOME_BUF_MAX = 512
POLICY_TEMPERATURE = 1.0
EMA_DECAY = 0.9

# -- 485g Fix (b): outcome-coupled, threat-conditioned, per-candidate driver --
OUTCOME_COUPLE_GAIN = 4.0   # gain on the per-candidate threat-weighted advantage
ADV_MIN_THRESHOLD = 1e-4    # relaxed (the per-candidate loss keys on harm-spread, not
                            # on a single high-advantage episode return)

# -- 485f Fix 1 (kept): defeat OFC clamp-saturation (485e railed at +/-0.1) --
OFC_BIAS_SCALE = 0.5         # raised 0.1 -> 0.5 so per-candidate variation lands in-band

# -- 485f Fix 2 (kept): SD-056 e2_action_contrastive in P0 -> action-divergent bank --
E2_CONTRASTIVE_WEIGHT = 0.05      # w_contrast on the InfoNCE term added to L_E2 recon
E2_CONTRASTIVE_TEMP = 0.1         # InfoNCE tau (SD-056 default)

# -- P2 behavioural-eval drive lengths (parallel to 485b/485c/485f) --
PRE_ONSET_TICKS = 25
POST_ONSET_TICKS = 40
CONTEXT_TICKS = 25

# -- Pre-registered thresholds (NOT derived from the run's own statistics) --
HEAD_DELTA_MIN = 1e-3        # 485d C2 statistic: head trained at all
BIAS_RANGE_FLOOR = 0.05      # 485g Fix (a): 1e-3 -> 0.05 == DEVAL_SHIFT_MARGIN
                             #   (re-aligned to the DV-significance floor so "ready"
                             #    can no longer certify a vacuous devaluation test)
DEVAL_SHIFT_MARGIN = 0.05    # TV(sel_high, sel_low) floor counting as a behavioural shift
SEPARATION_RATIO_MIN = 3.0   # between-context selection must exceed within-context jitter
BETWEEN_TV_FLOOR = 0.05      # absolute floor on between_context_tv (anti-vacuous C2)
FROZEN_SILENCE_EPS = 1e-9    # frozen head: bias==0 -> selection exactly uniform -> 0
MIN_PASS_SEEDS = 2


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
    """Mean pairwise L2 across the K candidate-bank z_world summaries. Informational
    diagnostic: SD-056 lifts this off zero (485f measured 0.017-0.043); the load-
    bearing readiness gate is the bias cross-candidate RANGE, not this."""
    if bank is None or bank.shape[0] < 2:
        return 0.0
    d = torch.cdist(bank, bank)
    K = bank.shape[0]
    eye = torch.eye(K, dtype=torch.bool, device=bank.device)
    return float(d[~eye].mean().item())


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
        # 485f Fix 2 (kept): SD-056 e2 action-conditional contrastive -> action-
        # divergent candidate bank. Applied in _encoder_step P0 over the same
        # real-transition batch as the recon MSE (in-batch-negatives InfoNCE).
        e2_action_contrastive_enabled=True,
        e2_action_contrastive_weight=E2_CONTRASTIVE_WEIGHT,
        e2_action_contrastive_temperature=E2_CONTRASTIVE_TEMP,
        e2_action_contrastive_min_batch_classes=2,
        # SD-033b OFC analog under test. ofc_harm_dim>0 -> z_harm enters state_code.
        use_ofc_analog=True,
        ofc_state_dim=16,
        ofc_harm_dim=HARM_DIM,
        ofc_bias_scale=OFC_BIAS_SCALE,        # 485f Fix 1 (kept): 0.1 -> 0.5
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
    # 485f Fix 1 contract: the raised clamp rail is wired through.
    assert abs(float(trainable.ofc.config.bias_scale) - OFC_BIAS_SCALE) < 1e-9
    # 485f Fix 2 contract: SD-056 contrastive helper is available + config armed.
    assert hasattr(trainable.e2, "world_forward_contrastive_loss")
    assert bool(getattr(trainable.config.e2, "e2_action_contrastive_enabled", False))
    # 485g Fix (a) contract: readiness floor re-aligned to the DV floor.
    assert abs(BIAS_RANGE_FLOOR - DEVAL_SHIFT_MARGIN) < 1e-9
    fz = frozen.ofc.state_bias_head[-1]
    tz = trainable.ofc.state_bias_head[-1]
    assert bool(torch.all(fz.weight == 0)) and bool(torch.all(fz.bias == 0))
    assert not (bool(torch.all(tz.weight == 0)) and bool(torch.all(tz.bias == 0)))
    assert len(list(trainable.ofc.bias_head_parameters())) == 4
    del frozen, trainable
    print(
        "Preflight PASS: OFC analog + harm_dim + train_state_bias_head flag "
        "+ bias_head_parameters + ofc_bias_scale=0.5 + SD-056 contrastive armed "
        "+ readiness floor 0.05 == DV floor + outcome-coupled driver wired",
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
# training (P0 encoder warmup + P1 head outcome-coupled REINFORCE-over-candidates)
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
            # 485f Fix 2 (kept): SD-056 action-contrastive auxiliary loss (in-batch-
            # negatives InfoNCE over the SAME real-transition batch) -> world_forward
            # preserves action divergence -> the proposer bank carries cross-candidate
            # z_world range. Returns 0 if < 2 distinct first-action classes (safe no-op).
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


def _ofc_outcome_coupled_range_loss(
    agent: REEAgent,
    outcome_buf: List[Tuple[torch.Tensor, int, float, float]],
    device,
) -> Tuple[torch.Tensor, int]:
    """485g Fix (b): outcome-coupled, threat-conditioned, PER-CANDIDATE REINFORCE-
    over-candidates (the SD-033a/598b pattern, sharpened to the OFC MECH-263
    devaluation function).

    For each sampled training snapshot (real candidate bank + the threat magnitude
    captured at collection time):
      harm_k = E3.harm_eval(bank_k)              # [K] per-candidate predicted harm
      adv_k  = gain * threat * (mean(harm_k) - harm_k)   # low-harm favoured, scaled
                                                 #   by the CURRENT threat magnitude
      bias   = ofc.compute_bias(bank)            # [K], gradient flows into the head
      loss  += -(adv_k * log_softmax(-bias)).sum()       # REINFORCE over ALL K

    Trains the head over ALL K candidates (the direct cross-candidate range driver)
    using an outcome-grounded (harm_eval), threat-conditioned advantage. Gradient is
    ~0 when threat ~ 0, so the head is only pushed to spread bias under threat --
    giving it the opportunity to learn threat-conditioned valuation (range under
    threat, flat when devalued) WITHOUT injecting the devaluation shift (the low-
    threat gradient is zero, not anti-range). Returns (loss, n_terms)."""
    if agent.ofc is None or len(outcome_buf) < 2:
        return torch.zeros(1, device=device), 0
    idxs = np.random.choice(
        len(outcome_buf), size=min(BATCH_SIZE, len(outcome_buf)), replace=False
    )
    terms: List[torch.Tensor] = []
    for i in idxs:
        snap, _sel_idx, _ep_return, threat = outcome_buf[int(i)]
        if snap.shape[0] < 2 or threat < ADV_MIN_THRESHOLD:
            continue
        with torch.no_grad():
            harm = agent.e3.harm_eval(snap).reshape(-1)  # [K] per-candidate harm cost
            if harm.numel() != snap.shape[0]:
                continue
            adv = OUTCOME_COUPLE_GAIN * float(threat) * (harm.mean() - harm)  # [K]
            if float(adv.abs().max()) < ADV_MIN_THRESHOLD:
                continue
        bias = agent.ofc.compute_bias(snap)  # [K], grad flows into the head
        log_p = F.log_softmax(-bias / POLICY_TEMPERATURE, dim=0)  # [K]
        terms.append(-(adv * log_p).sum())
    if not terms:
        return torch.zeros(1, device=device), 0
    return torch.stack(terms).mean(), len(terms)


def _p1_train(
    agent: REEAgent, env: CausalGridWorldV2, episodes: int, steps: int, arm: str, train_head: bool
) -> Dict:
    device = agent.device
    bias_opt = (
        optim.Adam(list(agent.ofc.bias_head_parameters()), lr=LR_OFC_BIAS)
        if train_head
        else None
    )
    head_init = _head_weight_vector(agent)
    grad_nonzero_updates = 0
    n_loss_terms = 0
    # outcome_buf entries: (bank_snapshot, selected_idx, ep_return, threat_at_snapshot)
    outcome_buf: List[Tuple[torch.Tensor, int, float, float]] = []
    baseline = 0.0
    bias_samples: List[float] = []
    agent.train()
    for ep in range(episodes):
        _, obs_dict = env.reset()
        agent.reset()
        ep_reward = 0.0
        ep_buf: List[Tuple[torch.Tensor, int, float]] = []
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
            snap = None
            threat = 0.0
            if step % RECORD_EVERY_N_STEPS == 0:
                snap = _build_snap(agent, candidates)
                # 485g: capture the CURRENT threat magnitude (z_harm norm) so the
                # outcome-coupled loss can be threat-conditioned. z_harm is the
                # sensory stream the OFC state_code reads (ofc_harm_dim>0).
                if latent.z_harm is not None:
                    threat = float(latent.z_harm.detach().norm().item())
            action = agent.select_action(candidates, ticks, temperature=1.0)
            if action is None:
                action = _action_to_onehot(
                    random.randint(0, env.action_dim - 1), env.action_dim, device
                )
                agent._last_action = action
            if snap is not None and action is not None:
                sel = 0
                aa = int(action.argmax(-1).item())
                for ci, c in enumerate(candidates[:N_PROBE_CANDIDATES]):
                    if getattr(c, "actions", None) is not None and c.actions.shape[1] >= 1:
                        if int(c.actions[:, 0, :].argmax(-1).item()) == aa:
                            sel = ci
                            break
                ep_buf.append((snap, sel, threat))
            if agent.ofc is not None:
                bias_samples.append(float(agent.ofc._last_bias_abs_mean))
            _, harm, done, _, obs_dict = env.step(action)
            ep_reward += float(harm)
            z_wp, z_sp, act_p = (
                latent.z_world.detach(),
                latent.z_self.detach(),
                action.detach(),
            )
            step += 1
            if done:
                break
        baseline = EMA_DECAY * baseline + (1.0 - EMA_DECAY) * ep_reward
        for snap, sel, threat in ep_buf:
            outcome_buf.append((snap, sel, ep_reward, threat))
        if len(outcome_buf) > OUTCOME_BUF_MAX:
            outcome_buf = outcome_buf[-OUTCOME_BUF_MAX:]
        if bias_opt is not None:
            l_loss, n_terms = _ofc_outcome_coupled_range_loss(agent, outcome_buf, device)
            n_loss_terms += n_terms
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
        "n_outcome_coupled_loss_terms": n_loss_terms,
    }


# --------------------------------------------------------------------------- #
# P2 behavioural eval
# --------------------------------------------------------------------------- #
def _collect_eval_context(
    agent: REEAgent, env: CausalGridWorldV2, steps: int
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Run the trained agent briefly and return (candidate_bank [K, world_dim],
    z_world_rep [1, world_dim]) -- a real candidate bank + a representative state for
    the controlled devaluation/discrimination drives."""
    device = agent.device
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
            action = agent.select_action(candidates, ticks, temperature=1.0)
            if action is None:
                action = _action_to_onehot(0, env.action_dim, device)
            _, _, done, _, obs_dict = env.step(action)
            if bank is not None:
                break
            if done:
                _, obs_dict = env.reset()
                agent.reset()
    return bank, z_world_rep


def _behavioural_eval(
    agent: REEAgent, bank: torch.Tensor, z_world_rep: torch.Tensor, seed: int
) -> Dict:
    """On the trained agent: measure devaluation-selection shift + task-role
    discrimination separation, all from the OFC head's induced candidate selection.
    Also measure the readiness positive control (trained-head bias cross-candidate
    range at the high-threat state) + the bank z_world spread diagnostic."""
    g = torch.Generator().manual_seed(20000 + seed)
    # Aversive outcome latents in the agent's z_harm domain (HARM_DIM).
    z_harm_high = torch.randn(1, HARM_DIM, generator=g) * 1.0   # threat present
    z_harm_low = torch.zeros(1, HARM_DIM)                        # threat removed (devalued)

    agent.eval()
    with torch.no_grad():
        # ---- P2-a devaluation sensitivity ----
        agent.ofc.reset()
        for _ in range(PRE_ONSET_TICKS):
            agent.ofc.update(z_world=z_world_rep, z_harm=z_harm_high, gate=1.0)
        sel_high = _selection_dist(agent, bank, POLICY_TEMPERATURE)
        bias_range_high = _bias_range(agent, bank)  # readiness positive control
        for _ in range(POST_ONSET_TICKS):
            agent.ofc.update(z_world=z_world_rep, z_harm=z_harm_low, gate=1.0)
        sel_low = _selection_dist(agent, bank, POLICY_TEMPERATURE)
        deval_shift = _tv(sel_high, sel_low)

        # ---- P2-b task-role discrimination ----
        # Two perceptually-matched task-role histories (z_world offset clusters)
        # converging on the same matched final input z_world_rep; jitter control A2.
        ctx = CONTEXT_TICKS
        hist_a = z_world_rep + 1.5 + 0.1 * torch.randn(ctx, WORLD_DIM, generator=g)
        hist_b = z_world_rep - 1.5 + 0.1 * torch.randn(ctx, WORLD_DIM, generator=g)
        hist_a2 = z_world_rep + 1.5 + 0.1 * torch.randn(ctx, WORLD_DIM, generator=g)

        def _drive_then_select(hist: torch.Tensor) -> torch.Tensor:
            agent.ofc.reset()
            for t in range(hist.shape[0]):
                agent.ofc.update(z_world=hist[t:t + 1], z_harm=None, gate=1.0)
            agent.ofc.update(z_world=z_world_rep, z_harm=None, gate=1.0)
            return _selection_dist(agent, bank, POLICY_TEMPERATURE)

        sel_a = _drive_then_select(hist_a)
        sel_b = _drive_then_select(hist_b)
        sel_a2 = _drive_then_select(hist_a2)
        between = _tv(sel_a, sel_b)
        within = _tv(sel_a, sel_a2)
        separation_ratio = between / max(within, 1e-6)

    return {
        "devaluation_selection_shift": deval_shift,
        "bias_range_high_threat": bias_range_high,
        "bank_zworld_spread": _bank_zworld_spread(bank),
        "between_context_selection_tv": between,
        "within_context_selection_jitter": within,
        "discrimination_separation_ratio": separation_ratio,
    }


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
        "bias_range_floor": BIAS_RANGE_FLOOR,
        "env_kwargs": ENV_KWARGS,
    }
    with arm_cell(seed, config_slice=full_config, script_path=Path(__file__)) as cell:
        agent, env = _make_agent(seed, train_head=train_head)
        for ep in range(p0):
            _encoder_step(agent, env, steps, arm, ep + 1)
        p1m = _p1_train(agent, env, p1, steps, arm, train_head)
        bank, z_world_rep = _collect_eval_context(agent, env, steps)
        if bank is None or z_world_rep is None:
            beh = {
                "devaluation_selection_shift": 0.0,
                "bias_range_high_threat": 0.0,
                "bank_zworld_spread": 0.0,
                "between_context_selection_tv": 0.0,
                "within_context_selection_jitter": 0.0,
                "discrimination_separation_ratio": 0.0,
                "eval_context_built": False,
            }
        else:
            beh = _behavioural_eval(agent, bank, z_world_rep, seed)
            beh["eval_context_built"] = True
        row = {
            "arm": arm,
            "seed": seed,
            "ofc_train_state_bias_head": train_head,
            "head_weight_delta_norm": p1m["head_weight_delta_norm"],
            "n_grad_nonzero_updates": p1m["n_grad_nonzero_updates"],
            "n_outcome_coupled_loss_terms": p1m["n_outcome_coupled_loss_terms"],
            "p1_mean_abs_ofc_bias": p1m["p1_mean_abs_ofc_bias"],
            **beh,
        }
        cell.stamp(row)
    print(
        f"verdict: {'PASS' if (train_head and beh.get('eval_context_built')) else 'FAIL'}  "
        f"head_delta={p1m['head_weight_delta_norm']:.6f}  "
        f"bias_range={beh['bias_range_high_threat']:.6f}  "
        f"bank_spread={beh['bank_zworld_spread']:.6f}  "
        f"deval_shift={beh['devaluation_selection_shift']:.4f}  "
        f"between_tv={beh['between_context_selection_tv']:.4f}  "
        f"sep_ratio={beh['discrimination_separation_ratio']:.3f}",
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

    # Readiness (non-vacuity): trained head trained AND trained-head bias has
    # cross-candidate RANGE >= BIAS_RANGE_FLOOR (now 0.05 == DV floor).
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

    return {
        "by_arm": by_arm,
        "n_seeds": n,
        "ready_seeds": ready_seeds,
        "readiness_met": readiness_met,
        "max_trained_head_delta": max((r["head_weight_delta_norm"] for r in trained), default=0.0),
        "max_trained_bias_range": max((r["bias_range_high_threat"] for r in trained), default=0.0),
        "max_trained_bank_spread": max((r.get("bank_zworld_spread", 0.0) for r in trained), default=0.0),
        "max_trained_between_tv": max((r["between_context_selection_tv"] for r in trained), default=0.0),
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
                "BIAS_RANGE_FLOOR -- 485g RE-ALIGNS this floor to 0.05 (== the "
                "DEVAL_SHIFT_MARGIN the load-bearing DVs require), so 'ready' can no "
                "longer certify a vacuous test the way 485f's 1e-3 floor did (485f "
                "cleared at 0.00898, ~50x below the DV floor). RANGE not magnitude. "
                "Below floor -> substrate_not_ready_requeue (stronger driver / budget), "
                "NEVER a SD-033b weakens."
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
                "HEAD_DELTA_MIN -- the head genuinely trained under the 485g outcome-"
                "coupled driver (485d C2 statistic)."
            ),
            "measured": float(result["max_trained_head_delta"]),
            "threshold": HEAD_DELTA_MIN,
            "control": "trainable-arm head trained on frozen-encoder state_code in P1",
            "met": result["max_trained_head_delta"] > HEAD_DELTA_MIN,
        },
    ]
    criteria_non_degenerate = {
        "C1": bool(acc["C1_devaluation_behavioural_shift"]) and bool(result["readiness_met"]),
        "C2": bool(acc["C2_discrimination_behavioural_separation"]) and bool(result["readiness_met"]),
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
        "max_trained_bank_zworld_spread": result["max_trained_bank_spread"],
        "max_trained_between_context_tv": result["max_trained_between_tv"],
        "arm_results": [r for arm in result["by_arm"].values() for r in arm],
        "substrate_under_test": "SD-033b train_state_bias_head + ofc_bias_scale=0.5 + SD-056 e2_action_contrastive P0 + outcome-coupled per-candidate driver",
        "unblocks": "commitment_closure:GAP-8 (SD-033b candidate -> provisional behavioural evidence)",
        "supersedes": "V3-EXQ-485f",
        "predecessor": "V3-EXQ-485f (non_contributory; readiness-gate miscalibration -- supersede)",
        "predecessors": [
            "V3-EXQ-485b (devaluation, representation-level)",
            "V3-EXQ-485c (task-role discrimination, representation-level)",
            "V3-EXQ-485d (trained-OFC-head substrate readiness)",
            "V3-EXQ-485e (trained-OFC-head behavioural; non_contributory range-starved)",
            "V3-EXQ-485f (trained-OFC-head behavioural; non_contributory readiness-gate miscalibration)",
        ],
        "notes": (
            "Evidence-grade trained-OFC-head BEHAVIOURAL validation of MECH-263 "
            "signatures a (devaluation sensitivity) + b (task-role discrimination), "
            "SUPERSEDING the non_contributory V3-EXQ-485f. 485f confirmed (via its "
            "manifest) that the candidate-bank z_world spread is NOT the binding "
            "constraint (bank_zworld_spread reached 0.043, SD-056 operative, 617 "
            "multistep readiness PASS); the OFC-head READOUT was the wall "
            "(bias_range 0.009 from a 0.043 input). 485g makes TWO fixes: (a) the "
            "readiness floor is re-aligned 1e-3 -> 0.05 (== DEVAL_SHIFT_MARGIN) so "
            "'ready' can no longer certify a vacuous DV; (b) the under-driving "
            "shared-return REINFORCE is replaced with an outcome-coupled, threat-"
            "conditioned, PER-CANDIDATE REINFORCE-over-candidates (the SD-033a/598b "
            "pattern sharpened to the OFC MECH-263 devaluation function: advantage = "
            "gain * threat * (mean_harm - per-candidate harm_eval), applied over all K "
            "candidates), with LR 5e-4 -> 2e-3 and P1 60 -> 120 ep. The driver gives "
            "the head outcome-grounded gradient to SPREAD bias under threat (the range "
            "driver) and ZERO gradient when devalued (threat ~ 0) -- giving it the "
            "opportunity to learn the threat-conditioned valuation C1 tests, WITHOUT "
            "injecting the devaluation shift (non-circular). Same-statistic non-"
            "vacuity gate: below 0.05 bias range -> substrate_not_ready_requeue, "
            "NEVER a false SD-033b weakens; readiness-met-but-DVs-fail -> honest "
            "weakens. OFC is the sole bias channel; modulatory-bias-selection-"
            "authority deliberately OFF; frozen-zeroed head is the GAP-1-analogue "
            "silence control."
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
    print("\n=== V3-EXQ-485g SUMMARY ===", flush=True)
    print(f"  readiness_met={result['readiness_met']} (ready_seeds={result['ready_seeds']}/{result['n_seeds']})", flush=True)
    print(f"  max_trained_bias_range={result['max_trained_bias_range']:.6f} (floor {BIAS_RANGE_FLOOR})", flush=True)
    print(f"  max_trained_bank_zworld_spread={result['max_trained_bank_spread']:.6f}", flush=True)
    print(f"  max_trained_between_context_tv={result['max_trained_between_tv']:.6f} (floor {BETWEEN_TV_FLOOR})", flush=True)
    print(f"  C1_devaluation_behavioural_shift={acc['C1_devaluation_behavioural_shift']} ({acc['n_c1_seeds']}/{result['n_seeds']})", flush=True)
    print(f"  C2_discrimination_behavioural_separation={acc['C2_discrimination_behavioural_separation']} ({acc['n_c2_seeds']}/{result['n_seeds']})", flush=True)
    print(f"  C3_frozen_head_silent={acc['C3_frozen_head_silent']} ({acc['n_c3_seeds']}/{result['n_seeds']})", flush=True)
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
