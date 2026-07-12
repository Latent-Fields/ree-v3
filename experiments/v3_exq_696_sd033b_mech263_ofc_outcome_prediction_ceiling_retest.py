#!/opt/local/bin/python3
"""V3-EXQ-696: SD-033b + MECH-263 OFC-analog outcome-prediction substrate-CEILING RETEST.

SUBSTRATE-LANDING retest of two substrate_ceiling claims:
  SD-033b -- OFC-analog (specific-outcome prediction + task-structure cognitive map):
    does enabling the OFC-analog specific-outcome-prediction channel (trained
    state_bias_head, z_harm-in-state_code) lift the previously-ceiling'd behavioural
    devaluation / discrimination metric STRICTLY above the OFF arm + a matched-noise
    control?
  MECH-263 -- OFC-analog state-space and specific-outcome representation (SD-033b
    represents BOTH state-space and specific-outcome). Signatures a (devaluation
    sensitivity) + b (same-sensory / different-task-role discrimination).

This is the latest OFC behavioural experiment in the 485-lineage (predecessor
V3-EXQ-485h, the disambiguating redesign). 696 re-derives that design as a
substrate_ceiling retest with a HARD P0 readiness / non-vacuity gate that
self-routes to substrate_not_ready_requeue (NEVER a substrate-verdict label) when
the OFC lifting channel is degenerate.

ENTANGLEMENT WARNING (load-bearing):
  Per REE_assembly/evidence/planning/substrate_queue.json entry
  f_dominance_conversion_ceiling, the OFC channel ("485h OFC") was *channel 1* and
  F-dominance the *channel 2* of an active conversion ceiling now mid-build
  (MECH-448 lever; falsifier V3-EXQ-689d queued). So a naive OFC retest may land on
  the mid-build ceiling. 696 therefore:
    (a) runs a P0 readiness/non-vacuity gate (p0_readiness_gate) whose `measured`
        statistic is the SAME RANGE statistic the load-bearing criterion routes on
        -- the trained-head OFC bias cross-candidate RANGE at the high-threat
        positive control, asserted on a POSITIVE control, NOT a magnitude/mean-abs
        proxy (the V3-EXQ-643 same-statistic rule). Below the floor (or oracle not
        ready) -> substrate_not_ready_requeue, non_contributory, NEVER a SD-033b /
        MECH-263 weakens;
    (b) emits a check_degeneracy() self-report over the load-bearing DV ranges so a
        pinned / vacuous criterion is flagged scoring_excluded="degenerate";
    (c) records the OFC bias RANGE at the devalued state + per task-role context
        (the T-vs-F disambiguator), plus an OPTIONAL live-E3 supplementary F-test
        (modulatory-bias-selection-authority ON, OFC bias as score_bias through the
        REAL selector). If the OFC bias reaches the accumulator but does NOT move
        committed selection (vs F-only AND high-vs-devalued), that is the positive-
        adjacent F-dominance reading -> 696 joins the MECH-439 / f_dominance
        conversion-ceiling cluster. This arm NEVER gates the verdict and NEVER drives
        a weakens.

DISCRIMINATIVE ARMS (matched seeds):
  ARM_0_OFF             -- use_ofc_analog=False (the previously-ceiling'd OFF arm).
  ARM_1_ON              -- use_ofc_analog=True with the trained state_bias_head
                           (ofc_train_state_bias_head=True, ofc_harm_dim>0) -- the
                           OFC lifting channel under test.
  ARM_2_MATCHED_NOISE   -- use_ofc_analog=True with a FROZEN-zeroed head
                           (ofc_train_state_bias_head=False); the GAP-1-analogue
                           silence control + matched-substrate-noise floor. The
                           load-bearing PASS requires ARM_1 to lift STRICTLY above
                           BOTH ARM_0 and ARM_2.

PHASED TRAINING (MANDATORY -- the head trains on encoder outputs):
  P0 encoder warmup (E1 + E2.world_forward + E3.harm_eval) WITH SD-056
     e2_action_contrastive armed -> action-divergent candidate bank.
  P1 frozen-encoder head train on the .detach()ed state_code via the paired
     threat-conditioned driver (spread@high-threat-state + anti-range@devalued-state),
     trained on BOTH regimes the eval tests.
  P2 BEHAVIOURAL eval.

PRE-REGISTERED ACCEPTANCE (per seed, then >= MIN_PASS_SEEDS of 3):
  READINESS (P0 non-vacuity precondition, SAME statistic the load-bearing criteria
    route on): ARM_1 trained-head bias cross-candidate RANGE at the high-threat
    positive control > BIAS_RANGE_FLOOR (0.05) AND head trained (weight-delta >
    floor). Below floor -> substrate_not_ready_requeue, NEVER a SD-033b/MECH-263 weakens.
  C1_devaluation_behavioural_shift (load-bearing): ARM_1 devaluation_selection_shift
    > DEVAL_SHIFT_MARGIN AND > ARM_0 shift + DEVAL_SHIFT_MARGIN AND > ARM_2 shift +
    DEVAL_SHIFT_MARGIN.
  C2_discrimination_behavioural_separation (load-bearing): ARM_1 separation_ratio >=
    SEPARATION_RATIO_MIN AND between_context_selection_tv >= BETWEEN_TV_FLOOR.
  C3_off_and_control_silent (load-bearing control): ARM_0 shift < 1e-9 AND ARM_2
    (frozen) shift < 1e-9 AND their between-context separations < 1e-9.
  PASS (supports SD-033b + MECH-263) = READINESS AND C1 AND C2 AND C3 on a majority.

INTERPRETATION GRID (one row per plausible outcome -> next action):
  PASS / supports: the OFC channel lifts the previously-ceiling'd behavioural
    diversity / outcome-prediction metric strictly above OFF + matched-noise. SD-033b
    / MECH-263 substrate_ceiling -> supports; closes the 485-lineage / unblocks GAP-8.
  FAIL / weakens (readiness MET, DVs fail): the trained head has genuine cross-
    candidate range (>= 0.05) but does NOT lift the metric over OFF + control despite
    the driver fix. Route /failure-autopsy with the recorded devalued-state range +
    live-E3 F-test to adjudicate (T) threat-invariant-bias vs (F) F-dominance
    conversion ceiling (MECH-439 / f_dominance_conversion_ceiling) BEFORE stamping.
    An honest result, NOT a vacuity artifact.
  FAIL / substrate_not_ready_requeue (readiness UNMET): the trained head still cannot
    drive bias range to the 0.05 DV floor (or the oracle is not ready) -> the
    behavioural test never ran. Re-queue with a stronger driver / larger budget. NOT
    a weakens.

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
from experiments._metrics import P0NotReady, check_degeneracy, p0_readiness_gate
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_696_sd033b_mech263_ofc_outcome_prediction_ceiling_retest"
EXPERIMENT_PURPOSE = "evidence"
QUEUE_ID = "V3-EXQ-696"
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

# -- paired threat-conditioned driver (kept from 485h Fix 2) --
OUTCOME_COUPLE_GAIN = 4.0   # gain on the per-candidate threat-weighted advantage
ADV_MIN_THRESHOLD = 1e-4    # min per-candidate harm spread for the spread term to fire
DEVALUED_ANTIRANGE_WEIGHT = 0.5   # weight on the devalued-state anti-range (variance) term
DRIVER_SNAP_BATCH = 16      # snapshots per P1 driver step (each runs 2 settle passes)

# -- defeat OFC clamp-saturation (kept from 485f/485h) --
OFC_BIAS_SCALE = 0.5

# -- SD-056 e2_action_contrastive in P0 -> action-divergent bank (kept from 485f/485h) --
E2_CONTRASTIVE_WEIGHT = 0.05
E2_CONTRASTIVE_TEMP = 0.1

# -- P2 behavioural-eval drive lengths --
PRE_ONSET_TICKS = 25
POST_ONSET_TICKS = 40
CONTEXT_TICKS = 25

# -- Pre-registered thresholds (NOT derived from the run's own statistics) --
HEAD_DELTA_MIN = 1e-3
BIAS_RANGE_FLOOR = 0.05      # == DEVAL_SHIFT_MARGIN (same-statistic non-vacuity gate)
DEVAL_SHIFT_MARGIN = 0.05
SEPARATION_RATIO_MIN = 3.0
BETWEEN_TV_FLOOR = 0.05
SILENCE_EPS = 1e-9
MIN_PASS_SEEDS = 2

# -- devalued-range collapse disambiguation thresholds --
DEVAL_RANGE_COLLAPSE_RATIO = 0.3   # devalued/high range below this -> range collapsed (-> F test)
THREAT_INVARIANT_RATIO = 0.6       # devalued/high range above this -> threat-invariant bias (-> T)

# -- live-E3 supplementary F-test (diagnostic-only) --
LIVE_E3_AUTHORITY_RANGE_FLOOR = 1e-4   # OFC bias must reach the accumulator with this range
LIVE_E3_COMMITTED_SHIFT_EPS = 1e-3     # committed-selection TV below this == "did not move"

ARMS: List[Tuple[str, str]] = [
    ("ARM_0_OFF", "off"),
    ("ARM_1_ON", "trained"),
    ("ARM_2_MATCHED_NOISE", "frozen"),
]


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
    isolation (REE lower-is-better -> lower bias = more favoured). When the OFC is
    OFF, compute_bias returns zeros -> a uniform distribution -> zero shift (the
    OFF silence control)."""
    if agent.ofc is None:
        k = bank.shape[0]
        return torch.full((k,), 1.0 / k)
    bias = agent.ofc.compute_bias(bank)  # [K], clamped to +/-ofc_bias_scale
    return F.softmax(-bias / temperature, dim=0).detach()


def _bias_range(agent: REEAgent, bank: torch.Tensor) -> float:
    """Cross-candidate range (max - min) of the OFC bias over the bank -- the
    statistic the load-bearing selection DVs route on. Zero when OFC is OFF."""
    if agent.ofc is None:
        return 0.0
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
    """Drive the OFC state_code toward a regime via gated EMA updates. No-op when OFC
    is OFF. reset=True zeros the state_code first (entry to the high-threat regime);
    reset=False continues from the current state_code (the high -> devalued transition,
    mirroring the eval's PRE_ONSET-then-POST_ONSET trajectory)."""
    if agent.ofc is None:
        return
    if reset:
        agent.ofc.reset()
    for _ in range(ticks):
        agent.ofc.update(z_world=z_world_rep, z_harm=z_harm, gate=1.0)


def _make_agent(seed: int, ofc_mode: str) -> Tuple[REEAgent, CausalGridWorldV2]:
    """ofc_mode: 'off' (use_ofc_analog=False), 'trained' (ON + trainable head),
    'frozen' (ON + zeroed/frozen head = the matched-noise silence control)."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    env = CausalGridWorldV2(seed=seed, **ENV_KWARGS)
    use_ofc = ofc_mode != "off"
    train_head = ofc_mode == "trained"
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
        # SD-056 e2 action-conditional contrastive (action-divergent bank).
        e2_action_contrastive_enabled=True,
        e2_action_contrastive_weight=E2_CONTRASTIVE_WEIGHT,
        e2_action_contrastive_temperature=E2_CONTRASTIVE_TEMP,
        e2_action_contrastive_min_batch_classes=2,
        # SD-033b OFC analog under test. ofc_harm_dim>0 -> z_harm enters state_code.
        use_ofc_analog=use_ofc,
        ofc_state_dim=16,
        ofc_harm_dim=HARM_DIM if use_ofc else 0,
        ofc_bias_scale=OFC_BIAS_SCALE,
        ofc_train_state_bias_head=train_head,
    )
    config.e3.commitment_threshold = 0.5
    config.heartbeat.beta_gate_bistable = True
    agent = REEAgent(config)
    return agent, env


def _preflight() -> None:
    off, _ = _make_agent(0, "off")
    trained, _ = _make_agent(2, "trained")
    frozen, _ = _make_agent(1, "frozen")
    assert off.ofc is None
    assert trained.ofc is not None and frozen.ofc is not None
    assert trained.ofc.config.train_state_bias_head is True
    assert frozen.ofc.config.train_state_bias_head is False
    assert trained.ofc.config.harm_dim == HARM_DIM
    assert abs(float(trained.ofc.config.bias_scale) - OFC_BIAS_SCALE) < 1e-9
    assert hasattr(trained.e2, "world_forward_contrastive_loss")
    assert bool(getattr(trained.config.e2, "e2_action_contrastive_enabled", False))
    # readiness floor re-aligned to the DV floor (same-statistic gate).
    assert abs(BIAS_RANGE_FLOOR - DEVAL_SHIFT_MARGIN) < 1e-9
    fz = frozen.ofc.state_bias_head[-1]
    tz = trained.ofc.state_bias_head[-1]
    assert bool(torch.all(fz.weight == 0)) and bool(torch.all(fz.bias == 0))
    assert not (bool(torch.all(tz.weight == 0)) and bool(torch.all(tz.bias == 0)))
    assert len(list(trained.ofc.bias_head_parameters())) == 4
    # live-E3 readout contract: E3Config carries the modulatory-authority flag.
    assert hasattr(trained.e3.config, "use_modulatory_selection_authority")
    del off, trained, frozen
    print(
        "Preflight PASS: OFF=None / trained-head ON / frozen-head matched-noise control "
        "+ harm_dim + bias_head_parameters + ofc_bias_scale=0.5 + SD-056 contrastive "
        "+ readiness floor 0.05 == DV floor + paired threat-conditioned driver "
        "+ modulatory-authority flag present",
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
    if agent.ofc is None:
        return torch.zeros(1)
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
            # SD-056 action-contrastive auxiliary loss.
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
    """PAIRED threat-conditioned, outcome-coupled, per-candidate driver (485h Fix 2).
    Trains the trained-OFC head on BOTH the high-threat regime (spread bias to favour
    low predicted harm) AND the devalued regime (suppress bias range), using a SHARED
    state_code construction (_settle_state_code) identical to the P2 eval.

    The anti-range term penalises bias-range MAGNITUDE at the devalued state_code -- it
    injects NO specific sel_low distribution or preferred candidate. The head must learn
    to read z_harm from the state_code to keep range high at z_harm_high and small at
    z_harm_low using ONLY the state_code difference (the SD-033b z_harm-in-state_code
    mechanism); that contrast can fail honestly. Returns (loss, n_spread, n_anti).

    Side effect: mutates ofc.state_code (settle passes); the caller saves/restores it."""
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
        if (train_head and agent.ofc is not None)
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
    [K, world_dim], z_world_rep [1, world_dim]) -- a real candidate list aligned with
    the bank + a representative state for the controlled devaluation / discrimination
    drives."""
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
    """Measure devaluation-selection shift + task-role discrimination separation from
    the OFC head's induced candidate selection. Also records the readiness positive
    control (bias range at the high-threat state) AND the bias range AT the devalued
    state + per task-role context (the T-vs-F disambiguator). On the OFF arm every OFC
    read returns zero -> the silence control."""
    agent.eval()
    with torch.no_grad():
        # ---- P2-a devaluation sensitivity ----
        _settle_state_code(agent, z_world_rep, z_harm_high, PRE_ONSET_TICKS, reset=True)
        sel_high = _selection_dist(agent, bank, POLICY_TEMPERATURE)
        bias_range_high = _bias_range(agent, bank)  # readiness positive control
        _settle_state_code(agent, z_world_rep, z_harm_low, POST_ONSET_TICKS, reset=False)
        sel_low = _selection_dist(agent, bank, POLICY_TEMPERATURE)
        bias_range_devalued = _bias_range(agent, bank)  # the disambiguator
        deval_shift = _tv(sel_high, sel_low)

        # ---- P2-b task-role discrimination ----
        ctx = CONTEXT_TICKS
        g = torch.Generator().manual_seed(20000 + seed)
        hist_a = z_world_rep + 1.5 + 0.1 * torch.randn(ctx, WORLD_DIM, generator=g)
        hist_b = z_world_rep - 1.5 + 0.1 * torch.randn(ctx, WORLD_DIM, generator=g)
        hist_a2 = z_world_rep + 1.5 + 0.1 * torch.randn(ctx, WORLD_DIM, generator=g)

        def _drive_then_select(hist: torch.Tensor) -> Tuple[torch.Tensor, float]:
            if agent.ofc is not None:
                agent.ofc.reset()
                for t in range(hist.shape[0]):
                    agent.ofc.update(z_world=hist[t:t + 1], z_harm=None, gate=1.0)
                agent.ofc.update(z_world=z_world_rep, z_harm=None, gate=1.0)
            sel = _selection_dist(agent, bank, POLICY_TEMPERATURE)
            rng = _bias_range(agent, bank)  # per-context bias range
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
    """OPTIONAL live-E3 supplementary F-test (diagnostic-only).

    Routes the OFC bias through the REAL E3 selector with modulatory-bias-selection-
    authority ON (OFC bias passed as score_bias so the authority rescales it into the
    F-dominated primary scores), and measures whether the COMMITTED selection shifts
    under devaluation, vs an F-only (no-OFC-bias) control.

    NON-VACUITY PRECONDITION: a verdict is only carried when the authority is genuinely
    active with a non-zero modulatory range at the high-threat state. This arm NEVER
    gates the SD-033b/MECH-263 verdict and NEVER drives a weakens.

    Verdict labels:
      live_e3_not_ready: authority did not engage (no F-test possible).
      f_dominance_positive_adjacent: OFC bias reached the accumulator but moved neither
        the high-vs-F-only nor the high-vs-devalued committed selection -> the (F)
        F-dominance reading; 696 joins the MECH-439 / f_dominance cluster.
      live_e3_devaluation_shift_present: committed selection DID shift under devaluation
        through the live selector (SD-033b-positive in the live loop).
      live_e3_inconclusive: otherwise."""
    out: Dict[str, object] = {
        "live_e3_status": "ok",
        "live_e3_label": "live_e3_not_ready",
        "live_e3_authority_active_high": False,
        "live_e3_authority_range_high": 0.0,
        "live_e3_committed_shift_devaluation": 0.0,
        "live_e3_committed_shift_vs_f_only": 0.0,
    }
    if agent.ofc is None or cands is None or bank is None or z_world_rep is None or len(cands) < 2:
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


def _empty_beh() -> Dict:
    return {
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


def run_arm_seed(arm: str, ofc_mode: str, seed: int, dry_run: bool) -> Dict:
    p0 = 3 if dry_run else P0_EPISODES
    p1 = 4 if dry_run else P1_EPISODES
    steps = 30 if dry_run else STEPS_PER_EPISODE
    train_head = ofc_mode == "trained"
    print(f"\nSeed {seed} Condition {arm} ofc_mode={ofc_mode}", flush=True)
    full_config = {
        "arm": arm,
        "ofc_mode": ofc_mode,
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
        agent, env = _make_agent(seed, ofc_mode=ofc_mode)
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
            beh = _empty_beh()
            live = {"live_e3_status": "no_eval_context", "live_e3_label": "live_e3_not_ready"}
        else:
            beh = _behavioural_eval(agent, bank, z_world_rep, z_harm_high, z_harm_low, seed)
            beh["eval_context_built"] = True
            # live-E3 F-test only on the trained head (ARM_1).
            if train_head:
                live = _live_e3_devaluation_readout(
                    agent, cands, bank, z_world_rep, z_harm_high, z_harm_low
                )
            else:
                live = {"live_e3_status": "non_trained_arm_skipped", "live_e3_label": "n/a"}
        row = {
            "arm": arm,
            "ofc_mode": ofc_mode,
            "seed": seed,
            "train_head": train_head,
            "head_weight_delta_norm": p1m["head_weight_delta_norm"],
            "n_grad_nonzero_updates": p1m["n_grad_nonzero_updates"],
            "n_spread_loss_terms": p1m["n_spread_loss_terms"],
            "n_antirange_loss_terms": p1m["n_antirange_loss_terms"],
            "p1_mean_abs_ofc_bias": p1m["p1_mean_abs_ofc_bias"],
            **beh,
            **live,
        }
        cell.stamp(row)
    verdict = "PASS" if (train_head and beh.get("eval_context_built")) else "FAIL"
    print(
        f"verdict: {verdict}  arm={arm}  "
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
    by_arm: Dict[str, List[Dict]] = {a[0]: [] for a in ARMS}
    for seed in seeds:
        for arm_label, ofc_mode in ARMS:
            by_arm[arm_label].append(run_arm_seed(arm_label, ofc_mode, seed, dry_run))

    off = by_arm["ARM_0_OFF"]
    trained = by_arm["ARM_1_ON"]
    control = by_arm["ARM_2_MATCHED_NOISE"]
    n = len(seeds)

    # READINESS over the trained arm: head trained AND high-threat bias RANGE > floor.
    ready_seeds = sum(
        1
        for r in trained
        if r["head_weight_delta_norm"] > HEAD_DELTA_MIN
        and r["bias_range_high_threat"] > BIAS_RANGE_FLOOR
        and r.get("eval_context_built", False)
    )
    readiness_met = ready_seeds >= MIN_PASS_SEEDS
    max_trained_bias_range = max((r["bias_range_high_threat"] for r in trained), default=0.0)
    max_trained_head_delta = max((r["head_weight_delta_norm"] for r in trained), default=0.0)

    # Per-seed load-bearing criteria (trained lifts strictly above OFF + control).
    c1_seeds = 0
    c2_seeds = 0
    c3_seeds = 0
    for fr, tr, ct in zip(off, trained, control):
        if (
            tr["devaluation_selection_shift"] > DEVAL_SHIFT_MARGIN
            and tr["devaluation_selection_shift"] > fr["devaluation_selection_shift"] + DEVAL_SHIFT_MARGIN
            and tr["devaluation_selection_shift"] > ct["devaluation_selection_shift"] + DEVAL_SHIFT_MARGIN
        ):
            c1_seeds += 1
        if (
            tr["discrimination_separation_ratio"] >= SEPARATION_RATIO_MIN
            and tr["between_context_selection_tv"] >= BETWEEN_TV_FLOOR
        ):
            c2_seeds += 1
        if (
            fr["devaluation_selection_shift"] < SILENCE_EPS
            and fr["between_context_selection_tv"] < SILENCE_EPS
            and ct["devaluation_selection_shift"] < SILENCE_EPS
            and ct["between_context_selection_tv"] < SILENCE_EPS
        ):
            c3_seeds += 1

    c1 = c1_seeds >= MIN_PASS_SEEDS
    c2 = c2_seeds >= MIN_PASS_SEEDS
    c3 = c3_seeds >= MIN_PASS_SEEDS
    overall_pass = readiness_met and c1 and c2 and c3

    # T-vs-F disambiguation summary over the trained arm.
    collapse_ratios = [
        r.get("devaluation_range_collapse_ratio", 0.0)
        for r in trained
        if r.get("eval_context_built", False)
    ]
    max_collapse_ratio = max(collapse_ratios, default=0.0)
    min_collapse_ratio = min(collapse_ratios, default=0.0)
    n_threat_invariant = sum(1 for x in collapse_ratios if x >= THREAT_INVARIANT_RATIO)
    n_range_collapsed = sum(1 for x in collapse_ratios if x <= DEVAL_RANGE_COLLAPSE_RATIO)
    live_labels = [r.get("live_e3_label", "n/a") for r in trained]
    n_f_dominance = sum(1 for x in live_labels if x == "f_dominance_positive_adjacent")
    n_live_ready = sum(1 for r in trained if r.get("live_e3_authority_active_high", False))

    return {
        "by_arm": by_arm,
        "n_seeds": n,
        "ready_seeds": ready_seeds,
        "readiness_met": readiness_met,
        "max_trained_head_delta": max_trained_head_delta,
        "max_trained_bias_range": max_trained_bias_range,
        "max_trained_bias_range_devalued": max(
            (r["bias_range_devalued"] for r in trained), default=0.0),
        "max_trained_bank_spread": max(
            (r.get("bank_zworld_spread", 0.0) for r in trained), default=0.0),
        "max_trained_between_tv": max(
            (r["between_context_selection_tv"] for r in trained), default=0.0),
        "max_collapse_ratio": max_collapse_ratio,
        "min_collapse_ratio": min_collapse_ratio,
        "n_threat_invariant_seeds": n_threat_invariant,
        "n_range_collapsed_seeds": n_range_collapsed,
        "n_f_dominance_seeds": n_f_dominance,
        "n_live_ready_seeds": n_live_ready,
        "acceptance": {
            "C1_devaluation_behavioural_shift": c1,
            "C2_discrimination_behavioural_separation": c2,
            "C3_off_and_control_silent": c3,
            "n_c1_seeds": c1_seeds,
            "n_c2_seeds": c2_seeds,
            "n_c3_seeds": c3_seeds,
            "pass": overall_pass,
        },
    }


def _degeneracy(result: Dict) -> Dict:
    """check_degeneracy() self-report over the load-bearing DV ranges (the same
    statistic the criteria route on). A pinned / vacuous criterion is flagged
    scoring_excluded=degenerate by the indexer."""
    trained = result["by_arm"]["ARM_1_ON"]
    return check_degeneracy({
        "trained_bias_range_high_threat": {
            "values": [r["bias_range_high_threat"] for r in trained],
            "floor": BIAS_RANGE_FLOOR / 2.0,
        },
        "trained_devaluation_selection_shift": [
            r["devaluation_selection_shift"] for r in trained
        ],
    })


def _disambiguation(result: Dict) -> Dict:
    """Record the T-vs-F disambiguation (NOT an auto-verdict). When readiness is met
    and C1 fails, the recorded devalued-state range + live-E3 F-test let governance /
    a follow-up /failure-autopsy route to (T) threat-invariant bias vs (F) F-dominance
    conversion ceiling (MECH-439 / f_dominance_conversion_ceiling)."""
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
            "Readiness-met + C1-fail is an honest result; route /failure-autopsy with "
            "bias_range_devalued + the live-E3 F-test to adjudicate T (threat-invariant "
            "bias) vs F (F-dominance conversion ceiling, MECH-439 / "
            "f_dominance_conversion_ceiling) before stamping a SD-033b/MECH-263 weakens."
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


def _readiness_preconditions(result: Dict) -> list:
    """P0 readiness/non-vacuity gate over the trained arm. The `measured` statistic is
    the SAME RANGE statistic the load-bearing criteria route on (a positive control),
    NOT a magnitude/mean-abs proxy (the V3-EXQ-643 same-statistic rule). Raises
    P0NotReady when unmet -> substrate_not_ready_requeue (NEVER a weakens)."""
    return p0_readiness_gate([
        {
            "name": "ofc_trained_head_bias_cross_candidate_range_supra_dv_floor",
            "measured": float(result["max_trained_bias_range"]),
            "threshold": BIAS_RANGE_FLOOR,
            "direction": "lower",
        },
        {
            "name": "ofc_head_weight_delta_supra_floor",
            "measured": float(result["max_trained_head_delta"]),
            "threshold": HEAD_DELTA_MIN,
            "direction": "lower",
        },
    ])


def _interpretation(result: Dict, preconditions: list, ready: bool) -> Dict:
    acc = result["acceptance"]
    if not ready:
        label = "substrate_not_ready_requeue"
    elif acc["pass"]:
        label = "sd033b_mech263_outcome_prediction_ceiling_lifted_supported"
    else:
        label = "sd033b_mech263_outcome_prediction_ceiling_persists"

    criteria = [
        {"name": "C1_devaluation_behavioural_shift", "load_bearing": True, "passed": bool(acc["C1_devaluation_behavioural_shift"])},
        {"name": "C2_discrimination_behavioural_separation", "load_bearing": True, "passed": bool(acc["C2_discrimination_behavioural_separation"])},
        {"name": "C3_off_and_control_silent", "load_bearing": True, "passed": bool(acc["C3_off_and_control_silent"])},
    ]
    criteria_non_degenerate = {
        "C1": bool(acc["C1_devaluation_behavioural_shift"]) and ready,
        "C2": bool(acc["C2_discrimination_behavioural_separation"]) and ready,
        "C3": bool(acc["C3_off_and_control_silent"]),
    }
    return {
        "label": label,
        "preconditions": preconditions,
        "criteria_non_degenerate": criteria_non_degenerate,
        "criteria": criteria,
        "disambiguation": _disambiguation(result),
    }


def _evidence_direction(result: Dict, ready: bool) -> str:
    if not ready:
        return "non_contributory"          # substrate_not_ready_requeue, NOT a weakens
    return "supports" if result["acceptance"]["pass"] else "weakens"


def write_manifest(result: Dict, dry_run: bool, elapsed: float) -> Tuple[Path, str]:
    acc = result["acceptance"]
    # P0 readiness/non-vacuity gate: self-routes to substrate_not_ready_requeue.
    try:
        preconditions = _readiness_preconditions(result)
        ready = True
    except P0NotReady as e:
        preconditions = e.preconditions
        ready = False
    interpretation = _interpretation(result, preconditions, ready)
    degeneracy = _degeneracy(result)
    direction = _evidence_direction(result, ready)
    outcome = "PASS" if (ready and acc["pass"]) else "FAIL"
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
        "timestamp_utc": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
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
        "non_degenerate": degeneracy["non_degenerate"],
        "degeneracy_reason": degeneracy["degeneracy_reason"],
        "degenerate_metrics": degeneracy["degenerate_metrics"],
        "readiness_met": ready,
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
        "substrate_under_test": (
            "SD-033b/MECH-263 OFC-analog outcome-prediction channel: train_state_bias_head "
            "+ ofc_harm_dim>0 + ofc_bias_scale=0.5 + SD-056 e2_action_contrastive P0 + paired "
            "threat-conditioned driver (spread@high + anti-range@devalued); ARM_0 OFF + "
            "ARM_2 frozen-zeroed matched-noise control"
        ),
        "unblocks": "SD-033b + MECH-263 substrate_ceiling -> supports; closes the 485-lineage / commitment_closure:GAP-8",
        "predecessor": "V3-EXQ-485h (the disambiguating-redesign behavioural validation in the 485 lineage)",
        "predecessors": [
            "V3-EXQ-485b/485c (representation-level diagnostics)",
            "V3-EXQ-485d (trained-OFC-head substrate readiness)",
            "V3-EXQ-485e/485f/485g (trained-OFC-head behavioural; non_contributory / substrate_ceiling)",
            "V3-EXQ-485h (disambiguating redesign; superseded 485g)",
        ],
        "conversion_ceiling_entanglement": (
            "Per substrate_queue f_dominance_conversion_ceiling, the OFC channel was channel 1 "
            "and F-dominance channel 2 of an active conversion ceiling mid-build (MECH-448 "
            "lever; falsifier V3-EXQ-689d queued). 696 self-routes substrate_not_ready_requeue "
            "(non_contributory, NEVER a weakens) if the OFC lifting channel is degenerate "
            "(trained-head bias cross-candidate RANGE below the 0.05 floor, the SAME statistic "
            "the load-bearing DVs route on). A readiness-met C1-fail records the devalued-state "
            "range + live-E3 F-test so governance can adjudicate (T) threat-invariant bias vs "
            "(F) the F-dominance conversion ceiling (MECH-439) before stamping a weakens."
        ),
        "notes": (
            "Evidence-grade substrate-landing RETEST of the SD-033b + MECH-263 substrate_ceiling "
            "claims: does the trained OFC-analog specific-outcome-prediction channel lift the "
            "previously-ceiling'd behavioural devaluation (MECH-263 signature a) + task-role "
            "discrimination (signature b) metric STRICTLY above the OFF arm AND the frozen-head "
            "matched-noise control? 3-arm ARM_0_OFF / ARM_1_ON / ARM_2_MATCHED_NOISE x 3 seeds. "
            "Same-statistic P0 non-vacuity gate (p0_readiness_gate): below the 0.05 trained-head "
            "bias-range floor (RANGE not magnitude) -> substrate_not_ready_requeue "
            "(non_contributory), NEVER a SD-033b/MECH-263 weakens. check_degeneracy self-report "
            "flags a pinned/vacuous criterion scoring_excluded=degenerate. The live-E3 "
            "modulatory-bias-selection-authority F-test is diagnostic-only and NEVER gates the "
            "verdict -- if the OFC bias reaches the accumulator but does not move committed "
            "selection, 696 joins the MECH-439 / f_dominance conversion-ceiling cluster. OFC is "
            "the sole bias channel in all arms; frozen-zeroed head is the GAP-1-analogue silence "
            "control."
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
    try:
        preconditions = _readiness_preconditions(result)
        ready = True
    except P0NotReady:
        ready = False
    interp = _interpretation(result, [], ready)
    print("\n=== V3-EXQ-696 SUMMARY ===", flush=True)
    print(f"  readiness_met={ready} (ready_seeds={result['ready_seeds']}/{result['n_seeds']})", flush=True)
    print(f"  max_trained_bias_range={result['max_trained_bias_range']:.6f} (floor {BIAS_RANGE_FLOOR})", flush=True)
    print(f"  max_trained_bias_range_devalued={result['max_trained_bias_range_devalued']:.6f}", flush=True)
    print(f"  collapse_ratio min/max={result['min_collapse_ratio']:.3f}/{result['max_collapse_ratio']:.3f}", flush=True)
    print(f"  C1_devaluation_behavioural_shift={acc['C1_devaluation_behavioural_shift']} ({acc['n_c1_seeds']}/{result['n_seeds']})", flush=True)
    print(f"  C2_discrimination_behavioural_separation={acc['C2_discrimination_behavioural_separation']} ({acc['n_c2_seeds']}/{result['n_seeds']})", flush=True)
    print(f"  C3_off_and_control_silent={acc['C3_off_and_control_silent']} ({acc['n_c3_seeds']}/{result['n_seeds']})", flush=True)
    print(f"  T-vs-F reading: {interp['disambiguation']['reading']}", flush=True)
    print(f"  live_e3 f_dominance_seeds={result['n_f_dominance_seeds']} ready_seeds={result['n_live_ready_seeds']}", flush=True)
    print(f"  interpretation: {interp['label']}", flush=True)
    print(f"  evidence_direction: {_evidence_direction(result, ready)}", flush=True)
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
