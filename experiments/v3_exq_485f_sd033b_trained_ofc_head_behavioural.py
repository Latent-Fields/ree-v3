#!/opt/local/bin/python3
"""V3-EXQ-485f: SD-033b trained-OFC-head BEHAVIOURAL validation (MECH-263 a+b).

commitment_closure:GAP-8 evidence-grade follow-on. SUCCESSOR (NOT supersede) to
V3-EXQ-485e, which self-routed substrate_not_ready_requeue -> non_contributory
(confirmed failure_autopsy_V3-EXQ-485e_2026-06-11). 485e trained the OFC head on
all 3 seeds but the cross-candidate bias RANGE -- the statistic every behavioural
DV routes on -- cleared the floor on only 1/3 seeds, so every selection DV
(devaluation shift, task-role separation) was vacuously ~0. The autopsy identified
TWO range-starving gaps + one measurement degeneracy; 485f fixes all three while
keeping the 485e design and the same-statistic non-vacuity readiness gate intact.

THREE FIXES (autopsy Section 8 routing a/b/c):
  (1) OFC CLAMP-SATURATION (the readiness FAIL on seeds 1 & 2). compute_bias clamps
      to +/-ofc_bias_scale; at the 485e value 0.1 the head's per-candidate output
      railed (p1_mean_abs_bias ~0.095-0.10 at the +/-0.1 rail), collapsing range to
      ~5e-4. FIX: raise ofc_bias_scale 0.1 -> 0.5 (OFC_BIAS_SCALE) so per-candidate
      variation lands in-band instead of railing. The 598b-style REINFORCE-over-
      candidates seeding the GAP-8 note cites is already the P1 training path here
      (_ofc_reinforce_loss); raising the rail is the missing calibration.
  (2) CANDIDATE-BANK z_world COLLAPSE (why even the READY 485e seed was vacuous).
      The bank is the proposer's e2.world_forward world_states[1], which under
      reconstruction-only training fits the action contribution toward zero
      (V3-EXQ-571 / ARC-065 GAP-A / SD-056) -> near-zero cross-candidate z_world
      spread -> even an unsaturated head reads near-identical per-candidate inputs.
      FIX: enable SD-056 e2_action_contrastive in P0 (landed but NOT enabled in the
      485e config). The P0 encoder warmup now adds an in-batch-negatives InfoNCE
      contrastive loss on world_forward (over the same real-transition batch used
      for the recon MSE), so world_forward preserves action divergence and the
      proposer bank carries action-divergent z_world range.
  (3) C2 MEASUREMENT DEGENERACY. 485e separation_ratio = between/max(within,1e-6)
      gave a 322-or-0 artefact (tiny/tiny); criteria_non_degenerate.C2=false caught
      it. FIX: add an absolute floor BETWEEN_TV_FLOOR on between_context_selection_tv
      so C2 cannot pass vacuously on a near-zero denominator.

DESIGN (unchanged from 485e -- 2-arm ARM_0_frozen_head / ARM_1_trainable_head; 3 seeds):
  OFF isolated (gated_policy / dACC / lateral_pfc / modulatory-bias-selection-
  authority all OFF) on the SD-054 bipartite reef/forage env, single varied factor
  ofc_train_state_bias_head, ofc_harm_dim>0 so z_harm enters the OFC state_code (the
  aversive-devaluation-capable shape). Phased training: P0 encoder warmup (now WITH
  SD-056 contrastive) -> P1 head REINFORCE on the frozen-encoder state_code -> P2
  BEHAVIOURAL eval. Behaviour = the candidate-selection distribution the OFC bias
  induces in isolation, selection[c] = softmax(-compute_bias(bank) / T) over a FIXED
  bank of real candidate first-step z_world summaries.

  P2-a DEVALUATION SENSITIVITY (MECH-263 a): drive the trained state_code with a
    stable HIGH-threat z_harm, read sel_high; DEVALUE (z_harm -> 0), read sel_low;
    devaluation_selection_shift = TV(sel_high, sel_low). Frozen head (bias==0) keeps
    a uniform selection -> shift exactly 0.
  P2-b TASK-ROLE DISCRIMINATION (MECH-263 b): two perceptually-matched task-role
    histories (z_world offset clusters A/B converging on the same matched final
    input), between = TV(sel_A, sel_B), within = TV(sel_A, sel_A2),
    separation_ratio = between / max(within, eps). Frozen head -> between == within == 0.

PRE-REGISTERED ACCEPTANCE (per seed, then >= MIN_PASS_SEEDS of 3):
  READINESS (non-vacuity precondition, SAME statistic the load-bearing criteria
    route on): the load-bearing DVs are selection differences, which can be non-zero
    ONLY if the trained-head bias carries cross-candidate RANGE. So readiness asserts,
    on a POSITIVE CONTROL (the high-threat state), the trained head's bias cross-
    candidate RANGE > BIAS_RANGE_FLOOR (range, NOT magnitude) AND the head actually
    trained (weight-delta-from-init > HEAD_DELTA_MIN). If readiness is unmet on a
    majority of seeds the substrate is not ready: the ONLY correct route is
    substrate_not_ready_requeue, NEVER a substrate verdict on SD-033b.
  C1_devaluation_behavioural_shift (load-bearing): ARM_1 devaluation_selection_shift
    > DEVAL_SHIFT_MARGIN AND > ARM_0 (frozen) shift + DEVAL_SHIFT_MARGIN.
  C2_discrimination_behavioural_separation (load-bearing): ARM_1 separation_ratio
    >= SEPARATION_RATIO_MIN AND between_context_selection_tv >= BETWEEN_TV_FLOOR
    (the 485f absolute-floor non-vacuity guard).
  C3_frozen_head_silent (load-bearing control): ARM_0 devaluation shift < 1e-9 AND
    ARM_0 between-context selection separation < 1e-9.
  PASS (supports SD-033b + MECH-263) = READINESS AND C1 AND C2 AND C3 on a majority.

INTERPRETATION GRID (one row per plausible outcome -> next action):
  PASS / supports: trained head produces devaluation-sensitive AND task-role-
    discriminative candidate selection that the frozen head does not. SD-033b
    candidate -> provisional behavioural evidence; closes commitment_closure:GAP-8.
  FAIL / weakens (readiness MET, DVs fail): the trained head's bias has genuine
    cross-candidate range but does NOT steer selection in a devaluation/discrimination
    way -> the OFC head does not carry the MECH-263 behavioural signature. Route
    /failure-autopsy before stamping; an honest weakens.
  FAIL / substrate_not_ready_requeue (readiness UNMET): the head did not train, or
    its bias is still clamp-saturated / the bank still collapsed (range below floor)
    -> the behavioural test never ran. Re-queue with further calibration (larger
    ofc_bias_scale / SD-056 weight / P0 budget). NOT a weakens.

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

EXPERIMENT_TYPE = "v3_exq_485f_sd033b_trained_ofc_head_behavioural"
EXPERIMENT_PURPOSE = "evidence"
QUEUE_ID = "V3-EXQ-485f"
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
P1_EPISODES = 60                       # evidence-grade head training (matches 485e)
STEPS_PER_EPISODE = 100
TRAIN_EPS = P0_EPISODES + P1_EPISODES  # progress denominator M

LR_E1 = 1e-3
LR_E2_WF = 1e-3
LR_E3_HARM = 1e-3
LR_OFC_BIAS = 5e-4
BATCH_SIZE = 32
WF_BUF_MAX = 256
N_PROBE_CANDIDATES = 8
RECORD_EVERY_N_STEPS = 4
OUTCOME_BUF_MAX = 512
POLICY_TEMPERATURE = 1.0
ADV_MIN_THRESHOLD = 0.005
EMA_DECAY = 0.9

# -- 485f Fix 1: defeat OFC clamp-saturation (485e railed at +/-0.1 on 2/3 seeds) --
OFC_BIAS_SCALE = 0.5         # raised 0.1 -> 0.5 so per-candidate variation lands in-band

# -- 485f Fix 2: SD-056 e2_action_contrastive in P0 -> action-divergent candidate bank --
E2_CONTRASTIVE_WEIGHT = 0.05      # w_contrast on the InfoNCE term added to L_E2 recon
E2_CONTRASTIVE_TEMP = 0.1         # InfoNCE tau (SD-056 default)

# -- P2 behavioural-eval drive lengths (parallel to 485b/485c) --
PRE_ONSET_TICKS = 25
POST_ONSET_TICKS = 40
CONTEXT_TICKS = 25

# -- Pre-registered thresholds (NOT derived from the run's own statistics) --
HEAD_DELTA_MIN = 1e-3        # 485d C2 statistic: head trained at all
BIAS_RANGE_FLOOR = 1e-3      # same-statistic readiness: trained-head bias cross-candidate RANGE
DEVAL_SHIFT_MARGIN = 0.05    # TV(sel_high, sel_low) floor counting as a behavioural shift
SEPARATION_RATIO_MIN = 3.0   # between-context selection must exceed within-context jitter by this
BETWEEN_TV_FLOOR = 0.05      # 485f Fix 3: absolute floor on between_context_tv (anti-vacuous C2)
FROZEN_SILENCE_EPS = 1e-9    # frozen head: bias==0 -> selection exactly uniform -> exactly 0
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
    """485f Fix-2 diagnostic: mean pairwise L2 across the K candidate-bank z_world
    summaries. Under the pre-SD-056 substrate this is ~0 (V3-EXQ-571 collapse);
    after SD-056 contrastive P0 it should carry action-divergent spread. Informational
    -- the load-bearing readiness gate stays the bias cross-candidate RANGE."""
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
        # 485f Fix 2: SD-056 e2 action-conditional contrastive -> action-divergent
        # candidate bank (the proposer's world_forward world_states[1]). Enabled in
        # the config for the audit trail; applied in _encoder_step P0 over the same
        # real-transition batch as the recon MSE (in-batch-negatives InfoNCE).
        e2_action_contrastive_enabled=True,
        e2_action_contrastive_weight=E2_CONTRASTIVE_WEIGHT,
        e2_action_contrastive_temperature=E2_CONTRASTIVE_TEMP,
        e2_action_contrastive_min_batch_classes=2,
        # SD-033b OFC analog under test. ofc_harm_dim>0 -> z_harm enters state_code.
        use_ofc_analog=True,
        ofc_state_dim=16,
        ofc_harm_dim=HARM_DIM,
        ofc_bias_scale=OFC_BIAS_SCALE,        # 485f Fix 1: raised 0.1 -> 0.5
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
    fz = frozen.ofc.state_bias_head[-1]
    tz = trainable.ofc.state_bias_head[-1]
    assert bool(torch.all(fz.weight == 0)) and bool(torch.all(fz.bias == 0))
    assert not (bool(torch.all(tz.weight == 0)) and bool(torch.all(tz.bias == 0)))
    assert len(list(trainable.ofc.bias_head_parameters())) == 4
    del frozen, trainable
    print(
        "Preflight PASS: OFC analog + harm_dim + train_state_bias_head flag "
        "+ bias_head_parameters + ofc_bias_scale=0.5 + SD-056 contrastive armed",
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
# training (P0 encoder warmup + P1 head REINFORCE) -- 485d machinery
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
            # 485f Fix 2: SD-056 action-contrastive auxiliary loss. In-batch-negatives
            # InfoNCE over the SAME real-transition batch (per-row z0/a/target): logits
            # [i,j] = -||world_forward(z0_j,a_j) - target_i||^2/tau, label i. Trains
            # world_forward to keep pred_b closer to its own next-state target than to
            # siblings' -> action divergence preserved -> the proposer bank carries
            # cross-candidate z_world range. Backprops into world_transition +
            # world_action_encoder (the e2_opt param set). Returns 0 if < 2 distinct
            # first-action classes in the batch (safe no-op).
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


def _ofc_reinforce_loss(
    agent: REEAgent, outcome_buf: List[Tuple[torch.Tensor, int, float]], baseline: float, device
) -> torch.Tensor:
    """REINFORCE on OFCAnalog.compute_bias -> state_bias_head (485d / 598b path)."""
    if agent.ofc is None or len(outcome_buf) < 2:
        return torch.zeros(1, device=device)
    idxs = np.random.choice(
        len(outcome_buf), size=min(BATCH_SIZE, len(outcome_buf)), replace=False
    )
    terms: List[torch.Tensor] = []
    for i in idxs:
        snap, sel_idx, ep_return = outcome_buf[int(i)]
        adv = ep_return - baseline
        if abs(adv) < ADV_MIN_THRESHOLD:
            continue
        bias = agent.ofc.compute_bias(snap)  # [K]
        log_p = F.log_softmax(-bias / POLICY_TEMPERATURE, dim=0)
        terms.append(-adv * log_p[min(sel_idx, bias.shape[0] - 1)])
    if not terms:
        return torch.zeros(1, device=device)
    return torch.stack(terms).mean()


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
    outcome_buf: List[Tuple[torch.Tensor, int, float]] = []
    baseline = 0.0
    bias_samples: List[float] = []
    agent.train()
    for ep in range(episodes):
        _, obs_dict = env.reset()
        agent.reset()
        ep_reward = 0.0
        ep_buf: List[Tuple[torch.Tensor, int]] = []
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
            if step % RECORD_EVERY_N_STEPS == 0:
                snap = _build_snap(agent, candidates)
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
                ep_buf.append((snap, sel))
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
        for snap, sel in ep_buf:
            outcome_buf.append((snap, sel, ep_reward))
        if len(outcome_buf) > OUTCOME_BUF_MAX:
            outcome_buf = outcome_buf[-OUTCOME_BUF_MAX:]
        if bias_opt is not None:
            l_loss = _ofc_reinforce_loss(agent, outcome_buf, baseline, device)
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
                f"  bias_abs={bias_samples[-1] if bias_samples else 0:.5f}",
                flush=True,
            )
    head_final = _head_weight_vector(agent)
    head_delta = float(torch.norm(head_final - head_init).item())
    return {
        "p1_mean_abs_ofc_bias": float(np.mean(bias_samples)) if bias_samples else 0.0,
        "head_weight_delta_norm": head_delta,
        "n_grad_nonzero_updates": grad_nonzero_updates,
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
    range at the high-threat state) + the 485f Fix-2 bank z_world spread diagnostic."""
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
    pre = 6 if dry_run else PRE_ONSET_TICKS
    post = 10 if dry_run else POST_ONSET_TICKS
    print(f"\nSeed {seed} Condition {arm} ofc_train_state_bias_head={train_head}", flush=True)
    full_config = {
        "arm": arm,
        "train_head": train_head,
        "p0": p0,
        "p1": p1,
        "steps": steps,
        "pre": pre,
        "post": post,
        "ofc_bias_scale": OFC_BIAS_SCALE,
        "e2_contrastive_weight": E2_CONTRASTIVE_WEIGHT,
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
    # frozen rows are paired to trained rows by seed order (same seeds list).
    n = len(seeds)

    # Readiness (non-vacuity): trained head trained AND trained-head bias has
    # cross-candidate RANGE (the statistic the load-bearing selection DVs route on).
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
        # C1 devaluation behavioural shift (trained beyond frozen control).
        if (
            tr["devaluation_selection_shift"] > DEVAL_SHIFT_MARGIN
            and tr["devaluation_selection_shift"]
            > fr["devaluation_selection_shift"] + DEVAL_SHIFT_MARGIN
        ):
            c1_seeds += 1
        # C2 discrimination behavioural separation -- 485f Fix 3: separation ratio
        # AND an absolute floor on between-context TV so a near-zero-denominator
        # ratio cannot pass vacuously.
        if (
            tr["discrimination_separation_ratio"] >= SEPARATION_RATIO_MIN
            and tr["between_context_selection_tv"] >= BETWEEN_TV_FLOOR
        ):
            c2_seeds += 1
        # C3 frozen-head silent control.
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

    # Readiness precondition: SAME statistic the load-bearing selection DVs route on
    # (cross-candidate bias RANGE -- a uniform bias gives a uniform softmax -> zero
    # shift/separation regardless of magnitude). Indexer recomputes met from
    # measured+threshold (floor direction).
    preconditions = [
        {
            "name": "ofc_trained_head_bias_cross_candidate_range_supra_floor",
            "description": (
                "ARM_1 trained-head OFC bias cross-candidate RANGE (max-min over the "
                "real candidate bank) at the high-threat positive-control state clears "
                "BIAS_RANGE_FLOOR -- the selection-distribution DVs (devaluation shift, "
                "task-role separation) can be non-zero only if the bias carries "
                "cross-candidate range. RANGE not magnitude (a clamp-saturated head has "
                "large |bias| but ~0 range). 485f raises ofc_bias_scale 0.1->0.5 to "
                "defeat the 485e clamp-saturation that starved this on 2/3 seeds."
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
                "HEAD_DELTA_MIN -- the head genuinely trained (485d C2 statistic)."
            ),
            "measured": float(result["max_trained_head_delta"]),
            "threshold": HEAD_DELTA_MIN,
            "control": "trainable-arm head trained on frozen-encoder state_code in P1",
            "met": result["max_trained_head_delta"] > HEAD_DELTA_MIN,
        },
    ]
    criteria_non_degenerate = {
        # C1 non-degenerate: the trained-head devaluation shift is read against the
        # frozen-head control (bias==0 -> exactly uniform selection), so a positive C1
        # is a genuine trained-vs-frozen contrast, not an artefact of both arms moving.
        "C1": bool(acc["C1_devaluation_behavioural_shift"]) and bool(result["readiness_met"]),
        # C2 non-degenerate: 485f gates separation on an absolute between-context TV
        # floor, so a passing C2 reflects genuine between-context selection separation,
        # not a near-zero-denominator ratio artefact (the 485e C2=false catch).
        "C2": bool(acc["C2_discrimination_behavioural_separation"]) and bool(result["readiness_met"]),
        # C3 non-degenerate: frozen control is genuinely silent (bias structurally 0).
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
        "substrate_under_test": "SD-033b train_state_bias_head + ofc_bias_scale=0.5 + SD-056 e2_action_contrastive P0",
        "unblocks": "commitment_closure:GAP-8 (SD-033b candidate -> provisional behavioural evidence)",
        "predecessor": "V3-EXQ-485e (non_contributory; substrate_not_ready_requeue -- successor, NOT supersede)",
        "predecessors": [
            "V3-EXQ-485b (devaluation, representation-level)",
            "V3-EXQ-485c (task-role discrimination, representation-level)",
            "V3-EXQ-485d (trained-OFC-head substrate readiness)",
            "V3-EXQ-485e (trained-OFC-head behavioural; non_contributory range-starved)",
        ],
        "notes": (
            "Evidence-grade trained-OFC-head BEHAVIOURAL validation of MECH-263 "
            "signatures a (devaluation sensitivity) + b (task-role discrimination), "
            "successor (NOT supersede) to the non_contributory V3-EXQ-485e. Three "
            "autopsy-routed fixes: (1) ofc_bias_scale 0.1->0.5 defeats OFC compute_bias "
            "clamp-saturation (485e railed at +/-0.1 on 2/3 seeds, range ~5e-4); (2) "
            "SD-056 e2_action_contrastive enabled in P0 (in-batch-negatives InfoNCE on "
            "world_forward over the recon batch) so the proposer candidate bank carries "
            "action-divergent z_world range (485e bank collapsed toward zero spread per "
            "V3-EXQ-571/ARC-065 GAP-A); (3) absolute between_context_tv floor on C2 so "
            "the separation ratio cannot pass vacuously on a near-zero denominator. "
            "Behaviour = OFC-bias-driven candidate-selection distribution (softmax over "
            "-compute_bias) in isolation; OFC is the sole bias channel; modulatory-bias-"
            "selection-authority deliberately OFF. Frozen-zeroed head (bias==0 -> uniform "
            "selection) is the GAP-1-analogue silence control. Same-statistic non-vacuity "
            "gate: trained-head bias cross-candidate RANGE on a positive control -> "
            "substrate_not_ready_requeue if below floor, never a false SD-033b weakens; "
            "readiness-met-but-DVs-fail -> honest weakens."
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
    print("\n=== V3-EXQ-485f SUMMARY ===", flush=True)
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
