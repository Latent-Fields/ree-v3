#!/opt/local/bin/python3
"""
V3-EXQ-186 -- Hybrid Benefit+Harm Selector (ARC-030 Paper-Ready)

Claims: ARC-030, MECH-112

Scientific question: Does combining observation-space benefit scoring
(ProximityEncoder + RFM direct proximity prediction) with E3 harm
avoidance (via a learned WorldForwardModel predicting z_world_next
for harm_eval) produce net resource collection lift?

This is the ARC-030 paper-ready result: approach + avoidance together
> either alone.

Design:
  4 conditions:
    COMBINED:     benefit_score + harm_score -> argmin combined_score
    BENEFIT_ONLY: argmax benefit_score (ProximityEncoder + RFM)
    HARM_ONLY:    argmin harm_score (WFM + E3 harm_eval)
    RANDOM:       uniform random

  WorldForwardModel (WFM): MLP predicting z_world_{t+1} from
    (z_world_t, action_onehot). Trained during warmup on MSE pairs.

  ProximityEncoder: 25->64->ReLU->16, proximity_head Linear(16,1),
    MSE on resource_proximity = 1/(1+manhattan_dist).

  ResourceForwardModel (RFM): (25+5)->64->ReLU->64->ReLU->25,
    MSE on rf_next prediction.

  Warmup: 400 episodes random actions (curriculum: first 80 eps
    _place_resource_near_agent). Trains all models simultaneously.

  Eval: 80 episodes per condition per seed.
  Seeds: [42, 7, 13]

PASS criteria:
  C1: benefit_ratio_combined_vs_random >= 1.3
  C2: benefit_ratio_combined_vs_harm_only >= 1.2
  C3: harm_rate_combined <= 1.3 * harm_rate_harm_only
  C4: benefit_rate_combined > benefit_rate_benefit_only (H-TRAP test)
"""

import sys
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_186_hybrid_benefit_harm_selector"
CLAIM_IDS = ["ARC-030", "MECH-112"]

RESOURCE_OBS_DIM = 25  # resource_field_view: 5x5 proximity grid
ENCODER_DIM = 16       # z_resource embedding dimension


# ------------------------------------------------------------------ #
# ProximityEncoder                                                     #
# ------------------------------------------------------------------ #

class ProximityEncoder(nn.Module):
    """
    Learns z_resource = f(resource_field_view) via proximity regression (MSE).
    Training target: resource_proximity = 1 / (1 + manhattan_dist_to_nearest).

    Smooth graded representation: contact ~ 1.0, 3 cells away ~ 0.25.
    Regression head (16 -> 1) forces encoder to learn distance-like features.
    Only encoder layers used for action selection scoring.
    """

    def __init__(self, resource_obs_dim: int = 25, encoder_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(resource_obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, encoder_dim),
        )
        self.proximity_head = nn.Linear(encoder_dim, 1)

    def encode(self, rf: torch.Tensor) -> torch.Tensor:
        """rf: [B, 25] -> z_resource: [B, ENCODER_DIM]"""
        return self.encoder(rf)

    def predict_proximity(self, rf: torch.Tensor) -> torch.Tensor:
        """rf: [B, 25] -> proximity_pred: [B, 1]  (trained to predict 0..1)"""
        return self.proximity_head(self.encoder(rf))


# ------------------------------------------------------------------ #
# ResourceForwardModel (raw rf space)                                  #
# ------------------------------------------------------------------ #

class ResourceForwardModel(nn.Module):
    """
    Predicts resource_field_view_next from (resource_field_view_curr, action).
    Operates in raw 25-dim observation space.
    """

    def __init__(self, resource_dim: int = 25, action_dim: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(resource_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, resource_dim),
        )

    def forward(self, rf: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([rf, action], dim=-1))


# ------------------------------------------------------------------ #
# WorldForwardModel (z_world space)                                    #
# ------------------------------------------------------------------ #

class WorldForwardModel(nn.Module):
    """
    Predicts z_world_{t+1} from (z_world_t, action_onehot).
    Used by HARM_ONLY and COMBINED conditions to project z_world
    forward for E3 harm_eval scoring.
    """

    def __init__(self, world_dim: int = 32, action_dim: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(world_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, world_dim),
        )

    def forward(self, z_world: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z_world, action], dim=-1))


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _ensure_2d(t: torch.Tensor) -> torch.Tensor:
    if t.dim() == 1:
        return t.unsqueeze(0)
    return t


def _place_resource_near_agent(env, max_dist: int = 3) -> bool:
    ax, ay = env.agent_x, env.agent_y
    candidates = []
    for dx in range(-max_dist, max_dist + 1):
        for dy in range(-max_dist, max_dist + 1):
            if dx == 0 and dy == 0:
                continue
            if abs(dx) + abs(dy) > max_dist:
                continue
            nx, ny = ax + dx, ay + dy
            if (0 < nx < env.size - 1 and 0 < ny < env.size - 1
                    and env.grid[nx, ny] == env.ENTITY_TYPES["empty"]):
                candidates.append((abs(dx) + abs(dy), nx, ny))
    if not candidates:
        return False
    candidates.sort()
    _, rx, ry = candidates[0]
    env.grid[rx, ry] = env.ENTITY_TYPES["resource"]
    env.resources.insert(0, [rx, ry])
    if env.use_proxy_fields:
        env._compute_proximity_fields()
    return True


def _resource_proximity(env) -> float:
    """1 / (1 + manhattan_dist_to_nearest_resource). 0.0 if no resources."""
    if not env.resources:
        return 0.0
    ax, ay = env.agent_x, env.agent_y
    min_dist = min(abs(ax - int(r[0])) + abs(ay - int(r[1])) for r in env.resources)
    return 1.0 / (1.0 + min_dist)


def _r2_score(preds: List[float], targets: List[float]) -> float:
    """Coefficient of determination R^2."""
    n = len(preds)
    if n < 4:
        return 0.0
    mean_t = sum(targets) / n
    ss_res = sum((p - t) ** 2 for p, t in zip(preds, targets))
    ss_tot = sum((t - mean_t) ** 2 for t in targets)
    if ss_tot < 1e-9:
        return 0.0
    return 1.0 - ss_res / ss_tot


# ------------------------------------------------------------------ #
# Action selection for each condition                                  #
# ------------------------------------------------------------------ #

def _select_action_combined(
    enc: ProximityEncoder,
    rfm: ResourceForwardModel,
    wfm: WorldForwardModel,
    agent: REEAgent,
    rf_curr: torch.Tensor,
    z_world_curr: torch.Tensor,
    num_actions: int,
    device,
    benefit_weight: float = 2.0,
    harm_weight: float = 1.0,
) -> int:
    """
    COMBINED: for each action a, compute:
      benefit_score = enc.predict_proximity(RFM(rf_curr, a))
      harm_score    = agent.e3.harm_eval(wfm(z_world_curr, a))
      combined_score = -benefit_weight * benefit_score + harm_weight * harm_score
    Pick action with LOWEST combined_score (high benefit, low harm).
    """
    with torch.no_grad():
        best_action = 0
        best_score = float("inf")
        for idx in range(num_actions):
            a_oh = _action_to_onehot(idx, num_actions, device)
            # Benefit: proximity prediction from RFM-predicted rf
            rf_pred = rfm(rf_curr, a_oh)
            benefit_score = enc.predict_proximity(rf_pred).mean().item()
            # Harm: E3 harm_eval on WFM-predicted z_world
            z_world_next = wfm(z_world_curr, a_oh)
            harm_score = agent.e3.harm_eval(z_world_next).mean().item()
            # Combined: minimize (-benefit + harm)
            score = -benefit_weight * benefit_score + harm_weight * harm_score
            if score < best_score:
                best_score = score
                best_action = idx
    return best_action


def _select_action_benefit_only(
    enc: ProximityEncoder,
    rfm: ResourceForwardModel,
    rf_curr: torch.Tensor,
    num_actions: int,
    device,
) -> int:
    """
    BENEFIT_ONLY: argmax_a enc.predict_proximity(RFM(rf_curr, a)).
    """
    with torch.no_grad():
        best_action = 0
        best_prox = -float("inf")
        for idx in range(num_actions):
            a_oh = _action_to_onehot(idx, num_actions, device)
            rf_pred = rfm(rf_curr, a_oh)
            prox = enc.predict_proximity(rf_pred).mean().item()
            if prox > best_prox:
                best_prox = prox
                best_action = idx
    return best_action


def _select_action_harm_only(
    wfm: WorldForwardModel,
    agent: REEAgent,
    z_world_curr: torch.Tensor,
    num_actions: int,
    device,
) -> int:
    """
    HARM_ONLY: argmin_a harm_eval(wfm(z_world_curr, a)).
    """
    with torch.no_grad():
        best_action = 0
        best_score = float("inf")
        for idx in range(num_actions):
            a_oh = _action_to_onehot(idx, num_actions, device)
            z_world_next = wfm(z_world_curr, a_oh)
            harm_score = agent.e3.harm_eval(z_world_next).mean().item()
            if harm_score < best_score:
                best_score = harm_score
                best_action = idx
    return best_action


# ------------------------------------------------------------------ #
# Main run function                                                    #
# ------------------------------------------------------------------ #

def _run_single(
    seed: int,
    warmup_episodes: int,
    eval_episodes: int,
    steps_per_episode: int,
    curriculum_episodes: int,
    self_dim: int,
    world_dim: int,
    lr: float,
    lr_enc: float,
    lr_rfm: float,
    lr_wfm: float,
    alpha_world: float,
    alpha_self: float,
    benefit_weight: float,
    harm_weight: float,
) -> Dict:
    torch.manual_seed(seed)
    random.seed(seed)

    env = CausalGridWorldV2(
        seed=seed,
        size=10,
        num_hazards=3,
        num_resources=4,
        hazard_harm=0.02,
        env_drift_interval=5,
        env_drift_prob=0.1,
        proximity_harm_scale=0.006,
        proximity_benefit_scale=0.006 * 0.6,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
        energy_decay=0.005,
        use_proxy_fields=True,
        resource_respawn_on_consume=True,
    )

    action_dim = env.action_dim

    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=alpha_world,
        alpha_self=alpha_self,
        reafference_action_dim=0,
        novelty_bonus_weight=0.0,
    )

    agent = REEAgent(config)
    enc = ProximityEncoder(resource_obs_dim=RESOURCE_OBS_DIM, encoder_dim=ENCODER_DIM)
    rfm = ResourceForwardModel(resource_dim=RESOURCE_OBS_DIM, action_dim=action_dim)
    wfm = WorldForwardModel(world_dim=world_dim, action_dim=action_dim)

    # Optimizers
    standard_params = [p for n, p in agent.named_parameters()
                       if "harm_eval_head" not in n]
    harm_eval_params = list(agent.e3.harm_eval_head.parameters())

    optimizer = optim.Adam(standard_params, lr=lr)
    harm_eval_opt = optim.Adam(harm_eval_params, lr=1e-4)
    enc_optimizer = optim.Adam(enc.parameters(), lr=lr_enc)
    rfm_optimizer = optim.Adam(rfm.parameters(), lr=lr_rfm)
    wfm_optimizer = optim.Adam(wfm.parameters(), lr=lr_wfm)

    # Replay buffers
    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []
    prox_buf_rf: List[torch.Tensor] = []
    prox_buf_tgt: List[float] = []
    wfm_buf_zw: List[torch.Tensor] = []      # z_world_t
    wfm_buf_act: List[torch.Tensor] = []     # action_t
    wfm_buf_zw_next: List[torch.Tensor] = [] # z_world_{t+1}
    MAX_BUF = 4000

    # Training diagnostics
    enc_train_losses: List[float] = []
    rfm_train_losses: List[float] = []
    wfm_train_losses: List[float] = []
    prox_preds_recent: List[float] = []
    prox_tgts_recent: List[float] = []

    counts: Dict[str, int] = {
        "hazard_approach": 0, "env_caused_hazard": 0,
        "agent_caused_hazard": 0, "none": 0, "resource": 0,
        "benefit_approach": 0,
    }

    # --- WARMUP TRAINING ---
    agent.train()
    enc.train()
    rfm.train()
    wfm.train()

    prev_rf: Optional[torch.Tensor] = None
    prev_action_oh: Optional[torch.Tensor] = None
    prev_z_world: Optional[torch.Tensor] = None

    for ep in range(warmup_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        prev_rf = None
        prev_action_oh = None
        prev_z_world = None

        if ep < curriculum_episodes:
            _place_resource_near_agent(env, max_dist=3)
            obs_dict = env._get_observation_dict()

        for step_i in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_harm = obs_dict.get("harm_obs", None)
            rf_curr = _ensure_2d(obs_dict["resource_field_view"].float())

            latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
            agent.clock.advance()
            z_world_curr = latent.z_world.detach()

            action_idx = random.randint(0, action_dim - 1)
            action_oh = _action_to_onehot(action_idx, action_dim, agent.device)
            agent._last_action = action_oh

            # Current resource proximity label (before step)
            prox_curr = _resource_proximity(env)

            _, harm_signal, done, info, obs_dict = env.step(action_oh)
            ttype = info.get("transition_type", "none")
            if ttype in counts:
                counts[ttype] += 1

            rf_next = _ensure_2d(obs_dict["resource_field_view"].float())

            # --- RFM training: predict rf_next from (rf_prev, action_prev) ---
            if prev_rf is not None and prev_action_oh is not None:
                rf_pred = rfm(prev_rf.detach(), prev_action_oh.detach())
                rfm_loss = F.mse_loss(rf_pred, rf_curr.detach())
                rfm_optimizer.zero_grad()
                rfm_loss.backward()
                torch.nn.utils.clip_grad_norm_(rfm.parameters(), 1.0)
                rfm_optimizer.step()
                rfm_train_losses.append(rfm_loss.item())

            # --- WFM training: predict z_world_next from (z_world_prev, action_prev) ---
            if prev_z_world is not None and prev_action_oh is not None:
                wfm_buf_zw.append(prev_z_world.detach())
                wfm_buf_act.append(prev_action_oh.detach())
                wfm_buf_zw_next.append(z_world_curr.detach())
                if len(wfm_buf_zw) > MAX_BUF:
                    wfm_buf_zw = wfm_buf_zw[-MAX_BUF:]
                    wfm_buf_act = wfm_buf_act[-MAX_BUF:]
                    wfm_buf_zw_next = wfm_buf_zw_next[-MAX_BUF:]

            if len(wfm_buf_zw) >= 16 and step_i % 4 == 0:
                k = min(32, len(wfm_buf_zw))
                indices = random.sample(range(len(wfm_buf_zw)), k)
                zw_batch = torch.cat([wfm_buf_zw[i] for i in indices], dim=0)
                act_batch = torch.cat([wfm_buf_act[i] for i in indices], dim=0)
                zw_next_batch = torch.cat([wfm_buf_zw_next[i] for i in indices], dim=0)
                zw_pred = wfm(zw_batch, act_batch)
                wfm_loss = F.mse_loss(zw_pred, zw_next_batch)
                wfm_optimizer.zero_grad()
                wfm_loss.backward()
                torch.nn.utils.clip_grad_norm_(wfm.parameters(), 1.0)
                wfm_optimizer.step()
                wfm_train_losses.append(wfm_loss.item())

            prev_rf = rf_curr.detach()
            prev_action_oh = action_oh.detach()
            prev_z_world = z_world_curr.detach()

            # --- Proximity regression training (ProximityEncoder) ---
            prox_buf_rf.append(rf_curr.detach())
            prox_buf_tgt.append(prox_curr)
            if len(prox_buf_rf) > MAX_BUF:
                prox_buf_rf = prox_buf_rf[-MAX_BUF:]
                prox_buf_tgt = prox_buf_tgt[-MAX_BUF:]

            if len(prox_buf_rf) >= 16:
                k = min(32, len(prox_buf_rf))
                indices = random.sample(range(len(prox_buf_rf)), k)
                rf_batch = torch.cat([prox_buf_rf[i] for i in indices], dim=0)
                tgt_batch = torch.tensor(
                    [prox_buf_tgt[i] for i in indices],
                    dtype=torch.float32,
                ).unsqueeze(1)
                pred_prox = enc.predict_proximity(rf_batch)
                enc_loss = F.mse_loss(pred_prox, tgt_batch)
                enc_optimizer.zero_grad()
                enc_loss.backward()
                torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
                enc_optimizer.step()
                enc_train_losses.append(enc_loss.item())
                # Track for R^2
                with torch.no_grad():
                    preds = pred_prox.squeeze(1).tolist()
                    tgts = tgt_batch.squeeze(1).tolist()
                prox_preds_recent.extend(preds)
                prox_tgts_recent.extend(tgts)
                if len(prox_preds_recent) > 2000:
                    prox_preds_recent = prox_preds_recent[-2000:]
                    prox_tgts_recent = prox_tgts_recent[-2000:]

            # --- Harm eval training (stratified BCE) ---
            is_harm = float(harm_signal) < 0
            if is_harm:
                harm_buf_pos.append(z_world_curr)
                if len(harm_buf_pos) > MAX_BUF // 2:
                    harm_buf_pos = harm_buf_pos[-(MAX_BUF // 2):]
            else:
                harm_buf_neg.append(z_world_curr)
                if len(harm_buf_neg) > MAX_BUF // 2:
                    harm_buf_neg = harm_buf_neg[-(MAX_BUF // 2):]

            if len(harm_buf_pos) >= 4 and len(harm_buf_neg) >= 4:
                k_pos = min(16, len(harm_buf_pos))
                k_neg = min(16, len(harm_buf_neg))
                pos_idx = torch.randperm(len(harm_buf_pos))[:k_pos].tolist()
                neg_idx = torch.randperm(len(harm_buf_neg))[:k_neg].tolist()
                zw_pos = torch.cat([harm_buf_pos[i] for i in pos_idx], dim=0)
                zw_neg = torch.cat([harm_buf_neg[i] for i in neg_idx], dim=0)
                zw_b = torch.cat([zw_pos, zw_neg], dim=0)
                target = torch.cat([
                    torch.ones(k_pos, 1, device=agent.device),
                    torch.zeros(k_neg, 1, device=agent.device),
                ], dim=0)
                pred_harm = agent.e3.harm_eval(zw_b)
                harm_loss = F.mse_loss(pred_harm, target)
                if harm_loss.requires_grad:
                    harm_eval_opt.zero_grad()
                    harm_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        agent.e3.harm_eval_head.parameters(), 0.5,
                    )
                    harm_eval_opt.step()

            # --- Standard agent training (E1 + E2) ---
            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total_loss = e1_loss + e2_loss
            if total_loss.requires_grad:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

            if done:
                break

        if (ep + 1) % 100 == 0 or ep == warmup_episodes - 1:
            enc_str = ""
            if enc_train_losses:
                recent = enc_train_losses[-100:]
                enc_str = f" enc_loss={sum(recent)/max(1,len(recent)):.4f}"
            rfm_str = ""
            if rfm_train_losses:
                recent = rfm_train_losses[-100:]
                rfm_str = f" rfm_loss={sum(recent)/max(1,len(recent)):.4f}"
            wfm_str = ""
            if wfm_train_losses:
                recent = wfm_train_losses[-100:]
                wfm_str = f" wfm_loss={sum(recent)/max(1,len(recent)):.4f}"
            prox_r2 = _r2_score(prox_preds_recent, prox_tgts_recent)
            curriculum_tag = " [curriculum]" if ep < curriculum_episodes else ""
            print(
                f"  [warmup] seed={seed}"
                f" ep {ep+1}/{warmup_episodes}"
                f" resource={counts['resource']}"
                f" prox_r2={prox_r2:.3f}"
                f"{enc_str}{rfm_str}{wfm_str}{curriculum_tag}",
                flush=True,
            )

    # Training diagnostics
    enc_final_loss = (
        float(sum(enc_train_losses[-500:]) / max(1, min(500, len(enc_train_losses))))
        if enc_train_losses else 0.0
    )
    rfm_final_loss = (
        float(sum(rfm_train_losses[-500:]) / max(1, min(500, len(rfm_train_losses))))
        if rfm_train_losses else 0.0
    )
    wfm_final_loss = (
        float(sum(wfm_train_losses[-500:]) / max(1, min(500, len(wfm_train_losses))))
        if wfm_train_losses else 0.0
    )
    prox_r2_final = _r2_score(prox_preds_recent, prox_tgts_recent)

    print(
        f"  [warmup done] seed={seed}"
        f" enc_loss={enc_final_loss:.4f}"
        f" rfm_loss={rfm_final_loss:.4f}"
        f" wfm_loss={wfm_final_loss:.4f}"
        f" prox_r2={prox_r2_final:.3f}",
        flush=True,
    )

    # ---- EVAL ----
    agent.eval()
    enc.eval()
    rfm.eval()
    wfm.eval()

    conditions = ["COMBINED", "BENEFIT_ONLY", "HARM_ONLY", "RANDOM"]
    cond_results: Dict[str, Dict] = {}

    for cond in conditions:
        benefit_per_ep: List[float] = []
        harm_per_ep: List[float] = []
        steps_per_ep_list: List[int] = []

        for eval_ep in range(eval_episodes):
            _, obs_dict = env.reset()
            agent.reset()
            ep_resources = 0
            ep_harm_sum = 0.0
            ep_steps = 0

            for step_i in range(steps_per_episode):
                obs_body = obs_dict["body_state"]
                obs_world = obs_dict["world_state"]
                obs_harm = obs_dict.get("harm_obs", None)
                rf_curr = _ensure_2d(obs_dict["resource_field_view"].float())

                with torch.no_grad():
                    latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
                    agent.clock.advance()
                    z_world_curr = latent.z_world.detach()

                if cond == "COMBINED":
                    action_idx = _select_action_combined(
                        enc, rfm, wfm, agent,
                        rf_curr, z_world_curr,
                        action_dim, agent.device,
                        benefit_weight=benefit_weight,
                        harm_weight=harm_weight,
                    )
                elif cond == "BENEFIT_ONLY":
                    action_idx = _select_action_benefit_only(
                        enc, rfm, rf_curr,
                        action_dim, agent.device,
                    )
                elif cond == "HARM_ONLY":
                    action_idx = _select_action_harm_only(
                        wfm, agent, z_world_curr,
                        action_dim, agent.device,
                    )
                else:  # RANDOM
                    action_idx = random.randint(0, action_dim - 1)

                action_oh = _action_to_onehot(action_idx, action_dim, agent.device)
                agent._last_action = action_oh

                _, harm_signal, done, info, obs_dict = env.step(action_oh)
                ttype = info.get("transition_type", "none")

                if ttype == "resource":
                    ep_resources += 1

                if float(harm_signal) < 0:
                    ep_harm_sum += abs(float(harm_signal))

                ep_steps += 1
                if done:
                    break

            benefit_per_ep.append(float(ep_resources))
            harm_per_ep.append(ep_harm_sum / max(1, ep_steps))
            steps_per_ep_list.append(ep_steps)

        benefit_rate = float(sum(benefit_per_ep)) / max(1, len(benefit_per_ep))
        harm_rate = float(sum(harm_per_ep)) / max(1, len(harm_per_ep))
        avg_steps = float(sum(steps_per_ep_list)) / max(1, len(steps_per_ep_list))

        cond_results[cond] = {
            "benefit_rate": benefit_rate,
            "harm_rate": harm_rate,
            "avg_steps": avg_steps,
        }

        print(
            f"  [eval] seed={seed} cond={cond}"
            f" benefit_rate={benefit_rate:.3f}"
            f" harm_rate={harm_rate:.5f}"
            f" avg_steps={avg_steps:.1f}",
            flush=True,
        )

    return {
        "seed": seed,
        "conditions": cond_results,
        "enc_final_loss": enc_final_loss,
        "rfm_final_loss": rfm_final_loss,
        "wfm_final_loss": wfm_final_loss,
        "prox_r2": prox_r2_final,
        "train_resource_events": counts["resource"],
    }


# ------------------------------------------------------------------ #
# Aggregation and output                                               #
# ------------------------------------------------------------------ #

def run(
    seeds: Tuple = (42, 7, 13),
    warmup_episodes: int = 400,
    eval_episodes: int = 80,
    steps_per_episode: int = 200,
    curriculum_episodes: int = 80,
    self_dim: int = 16,
    world_dim: int = 32,
    lr: float = 1e-4,
    lr_enc: float = 1e-3,
    lr_rfm: float = 5e-4,
    lr_wfm: float = 5e-4,
    alpha_world: float = 0.9,
    alpha_self: float = 0.3,
    benefit_weight: float = 2.0,
    harm_weight: float = 1.0,
    **kwargs,
) -> dict:
    all_results: List[Dict] = []

    for seed in seeds:
        print(
            f"\n[V3-EXQ-186] seed={seed}"
            f" warmup={warmup_episodes} eval={eval_episodes}"
            f" steps={steps_per_episode}"
            f" benefit_weight={benefit_weight}"
            f" harm_weight={harm_weight}",
            flush=True,
        )
        r = _run_single(
            seed=seed,
            warmup_episodes=warmup_episodes,
            eval_episodes=eval_episodes,
            steps_per_episode=steps_per_episode,
            curriculum_episodes=curriculum_episodes,
            self_dim=self_dim,
            world_dim=world_dim,
            lr=lr,
            lr_enc=lr_enc,
            lr_rfm=lr_rfm,
            lr_wfm=lr_wfm,
            alpha_world=alpha_world,
            alpha_self=alpha_self,
            benefit_weight=benefit_weight,
            harm_weight=harm_weight,
        )
        all_results.append(r)

    def _avg_cond(key: str, cond: str) -> float:
        vals = [r["conditions"][cond][key] for r in all_results]
        return float(sum(vals) / max(1, len(vals)))

    def _avg_key(key: str) -> float:
        vals = [r[key] for r in all_results]
        return float(sum(vals) / max(1, len(vals)))

    # Aggregate per condition
    combined_benefit_rate = _avg_cond("benefit_rate", "COMBINED")
    combined_harm_rate = _avg_cond("harm_rate", "COMBINED")

    benefit_only_benefit_rate = _avg_cond("benefit_rate", "BENEFIT_ONLY")
    benefit_only_harm_rate = _avg_cond("harm_rate", "BENEFIT_ONLY")

    harm_only_benefit_rate = _avg_cond("benefit_rate", "HARM_ONLY")
    harm_only_harm_rate = _avg_cond("harm_rate", "HARM_ONLY")

    random_benefit_rate = _avg_cond("benefit_rate", "RANDOM")
    random_harm_rate = _avg_cond("harm_rate", "RANDOM")

    avg_enc_loss = _avg_key("enc_final_loss")
    avg_rfm_loss = _avg_key("rfm_final_loss")
    avg_wfm_loss = _avg_key("wfm_final_loss")
    avg_prox_r2 = _avg_key("prox_r2")

    # Ratios
    benefit_ratio_vs_random = (
        combined_benefit_rate / max(1e-6, random_benefit_rate)
        if random_benefit_rate > 1e-6 else 0.0
    )
    benefit_ratio_vs_harm_only = (
        combined_benefit_rate / max(1e-6, harm_only_benefit_rate)
        if harm_only_benefit_rate > 1e-6 else 0.0
    )
    harm_ratio_vs_harm_only = (
        combined_harm_rate / max(1e-6, harm_only_harm_rate)
        if harm_only_harm_rate > 1e-6 else 0.0
    )

    # PASS criteria
    c1_pass = benefit_ratio_vs_random >= 1.3
    c2_pass = benefit_ratio_vs_harm_only >= 1.2
    c3_pass = combined_harm_rate <= 1.3 * harm_only_harm_rate
    c4_pass = combined_benefit_rate > benefit_only_benefit_rate

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass])
    status = "PASS" if all_pass else "FAIL"

    if all_pass:
        decision = "retain_ree"
    elif criteria_met >= 3:
        decision = "hybridize"
    else:
        decision = "retire_ree_claim"

    print(f"\n[V3-EXQ-186] Final results:", flush=True)
    print(
        f"  COMBINED:     benefit_rate={combined_benefit_rate:.3f}"
        f"  harm_rate={combined_harm_rate:.5f}",
        flush=True,
    )
    print(
        f"  BENEFIT_ONLY: benefit_rate={benefit_only_benefit_rate:.3f}"
        f"  harm_rate={benefit_only_harm_rate:.5f}",
        flush=True,
    )
    print(
        f"  HARM_ONLY:    benefit_rate={harm_only_benefit_rate:.3f}"
        f"  harm_rate={harm_only_harm_rate:.5f}",
        flush=True,
    )
    print(
        f"  RANDOM:       benefit_rate={random_benefit_rate:.3f}"
        f"  harm_rate={random_harm_rate:.5f}",
        flush=True,
    )
    print(
        f"  benefit_ratio_vs_random={benefit_ratio_vs_random:.2f}x"
        f"  benefit_ratio_vs_harm_only={benefit_ratio_vs_harm_only:.2f}x"
        f"  harm_ratio_vs_harm_only={harm_ratio_vs_harm_only:.2f}x",
        flush=True,
    )
    print(
        f"  enc_loss={avg_enc_loss:.4f}"
        f"  rfm_loss={avg_rfm_loss:.4f}"
        f"  wfm_loss={avg_wfm_loss:.4f}"
        f"  prox_r2={avg_prox_r2:.3f}",
        flush=True,
    )
    print(
        f"  decision={decision}  status={status} ({criteria_met}/4)",
        flush=True,
    )

    failure_notes: List[str] = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: benefit_ratio_vs_random={benefit_ratio_vs_random:.2f}x < 1.3x."
            f" Combined does not outperform random baseline."
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: benefit_ratio_vs_harm_only={benefit_ratio_vs_harm_only:.2f}x < 1.2x."
            f" Benefit channel not adding enough over harm-only avoidance."
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: combined_harm_rate={combined_harm_rate:.5f}"
            f" > 1.3 * harm_only_harm_rate={harm_only_harm_rate:.5f}."
            f" Goal pursuit increases harm beyond acceptable margin."
        )
    if not c4_pass:
        failure_notes.append(
            f"C4 FAIL (H-TRAP): combined_benefit_rate={combined_benefit_rate:.3f}"
            f" <= benefit_only_benefit_rate={benefit_only_benefit_rate:.3f}."
            f" Harm avoidance does NOT improve net benefit -- no H-TRAP effect."
        )

    for note in failure_notes:
        print(f"  {note}", flush=True)

    per_seed_rows = "\n".join(
        f"  seed={r['seed']}:"
        f" combined_br={r['conditions']['COMBINED']['benefit_rate']:.3f}"
        f" benefit_only_br={r['conditions']['BENEFIT_ONLY']['benefit_rate']:.3f}"
        f" harm_only_br={r['conditions']['HARM_ONLY']['benefit_rate']:.3f}"
        f" random_br={r['conditions']['RANDOM']['benefit_rate']:.3f}"
        f" prox_r2={r['prox_r2']:.3f}"
        for r in all_results
    )

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(
            f"- {n}" for n in failure_notes
        )

    summary_markdown = (
        f"# V3-EXQ-186 -- Hybrid Benefit+Harm Selector (ARC-030)\n\n"
        f"**Status:** {status}\n"
        f"**Claims:** ARC-030, MECH-112\n"
        f"**Decision:** {decision}\n"
        f"**Seeds:** {list(seeds)}\n\n"
        f"## Design\n\n"
        f"4 conditions: COMBINED (benefit+harm), BENEFIT_ONLY (ProximityEncoder+RFM),\n"
        f"HARM_ONLY (WFM+E3 harm_eval), RANDOM.\n\n"
        f"COMBINED action selection:\n"
        f"  combined_score = -benefit_weight * enc.predict_proximity(RFM(rf, a))\n"
        f"                   + harm_weight * harm_eval(wfm(z_world, a))\n"
        f"  Pick action with lowest combined_score.\n\n"
        f"benefit_weight={benefit_weight}, harm_weight={harm_weight}\n\n"
        f"## Results\n\n"
        f"| Condition | benefit_rate | harm_rate |\n"
        f"|---|---|---|\n"
        f"| COMBINED | {combined_benefit_rate:.3f} | {combined_harm_rate:.5f} |\n"
        f"| BENEFIT_ONLY | {benefit_only_benefit_rate:.3f}"
        f" | {benefit_only_harm_rate:.5f} |\n"
        f"| HARM_ONLY | {harm_only_benefit_rate:.3f}"
        f" | {harm_only_harm_rate:.5f} |\n"
        f"| RANDOM | {random_benefit_rate:.3f} | {random_harm_rate:.5f} |\n\n"
        f"**benefit_ratio_vs_random:** {benefit_ratio_vs_random:.2f}x\n"
        f"**benefit_ratio_vs_harm_only:** {benefit_ratio_vs_harm_only:.2f}x\n"
        f"**harm_ratio_vs_harm_only:** {harm_ratio_vs_harm_only:.2f}x\n\n"
        f"## Training Diagnostics\n\n"
        f"| Model | Final Loss |\n"
        f"|---|---|\n"
        f"| ProximityEncoder | {avg_enc_loss:.4f} |\n"
        f"| ResourceForwardModel | {avg_rfm_loss:.4f} |\n"
        f"| WorldForwardModel | {avg_wfm_loss:.4f} |\n\n"
        f"**prox_r2 (avg):** {avg_prox_r2:.3f}\n\n"
        f"## PASS Criteria\n\n"
        f"| Criterion | Result | Value |\n"
        f"|---|---|---|\n"
        f"| C1: combined/random >= 1.3x | {'PASS' if c1_pass else 'FAIL'}"
        f" | {benefit_ratio_vs_random:.2f}x |\n"
        f"| C2: combined/harm_only >= 1.2x | {'PASS' if c2_pass else 'FAIL'}"
        f" | {benefit_ratio_vs_harm_only:.2f}x |\n"
        f"| C3: combined_harm <= 1.3*harm_only | {'PASS' if c3_pass else 'FAIL'}"
        f" | {harm_ratio_vs_harm_only:.2f}x |\n"
        f"| C4: combined > benefit_only (H-TRAP) | {'PASS' if c4_pass else 'FAIL'}"
        f" | {combined_benefit_rate:.3f} vs {benefit_only_benefit_rate:.3f} |\n\n"
        f"Criteria met: {criteria_met}/4 -> **{status}**\n\n"
        f"## Per-Seed\n\n{per_seed_rows}\n"
        f"{failure_section}\n"
    )

    metrics = {
        "combined_benefit_rate":       float(combined_benefit_rate),
        "combined_harm_rate":          float(combined_harm_rate),
        "benefit_only_benefit_rate":   float(benefit_only_benefit_rate),
        "benefit_only_harm_rate":      float(benefit_only_harm_rate),
        "harm_only_benefit_rate":      float(harm_only_benefit_rate),
        "harm_only_harm_rate":         float(harm_only_harm_rate),
        "random_benefit_rate":         float(random_benefit_rate),
        "random_harm_rate":            float(random_harm_rate),
        "benefit_ratio_vs_random":     float(benefit_ratio_vs_random),
        "benefit_ratio_vs_harm_only":  float(benefit_ratio_vs_harm_only),
        "harm_ratio_vs_harm_only":     float(harm_ratio_vs_harm_only),
        "enc_final_loss":              float(avg_enc_loss),
        "rfm_final_loss":              float(avg_rfm_loss),
        "wfm_final_loss":              float(avg_wfm_loss),
        "prox_r2":                     float(avg_prox_r2),
        "benefit_weight":              float(benefit_weight),
        "harm_weight":                 float(harm_weight),
        "n_seeds":                     float(len(seeds)),
        "crit1_pass": 1.0 if c1_pass else 0.0,
        "crit2_pass": 1.0 if c2_pass else 0.0,
        "crit3_pass": 1.0 if c3_pass else 0.0,
        "crit4_pass": 1.0 if c4_pass else 0.0,
        "criteria_met": float(criteria_met),
    }

    per_seed_results = []
    for r in all_results:
        per_seed_results.append({
            "seed": r["seed"],
            "combined_benefit_rate": r["conditions"]["COMBINED"]["benefit_rate"],
            "combined_harm_rate": r["conditions"]["COMBINED"]["harm_rate"],
            "benefit_only_benefit_rate": r["conditions"]["BENEFIT_ONLY"]["benefit_rate"],
            "benefit_only_harm_rate": r["conditions"]["BENEFIT_ONLY"]["harm_rate"],
            "harm_only_benefit_rate": r["conditions"]["HARM_ONLY"]["benefit_rate"],
            "harm_only_harm_rate": r["conditions"]["HARM_ONLY"]["harm_rate"],
            "random_benefit_rate": r["conditions"]["RANDOM"]["benefit_rate"],
            "random_harm_rate": r["conditions"]["RANDOM"]["harm_rate"],
            "prox_r2": r["prox_r2"],
            "enc_final_loss": r["enc_final_loss"],
            "rfm_final_loss": r["rfm_final_loss"],
            "wfm_final_loss": r["wfm_final_loss"],
        })

    return {
        "status": status,
        "metrics": metrics,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "per_seed_results": per_seed_results,
        "evidence_direction": (
            "supports" if all_pass
            else ("mixed" if criteria_met >= 2 else "weakens")
        ),
        "experiment_type": EXPERIMENT_TYPE,
        "fatal_error_count": 0,
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",           type=int,   nargs="+", default=[42, 7, 13])
    parser.add_argument("--warmup",          type=int,   default=400)
    parser.add_argument("--eval-eps",        type=int,   default=80)
    parser.add_argument("--steps",           type=int,   default=200)
    parser.add_argument("--curriculum",      type=int,   default=80)
    parser.add_argument("--alpha-world",     type=float, default=0.9)
    parser.add_argument("--alpha-self",      type=float, default=0.3)
    parser.add_argument("--lr",              type=float, default=1e-4)
    parser.add_argument("--lr-enc",          type=float, default=1e-3)
    parser.add_argument("--lr-rfm",          type=float, default=5e-4)
    parser.add_argument("--lr-wfm",          type=float, default=5e-4)
    parser.add_argument("--benefit-weight",  type=float, default=2.0)
    parser.add_argument("--harm-weight",     type=float, default=1.0)
    parser.add_argument("--dry-run",         action="store_true",
                        help="Run 1 seed, 3 warmup eps, 3 eval eps for smoke test")
    args = parser.parse_args()

    if args.dry_run:
        args.seeds = [42]
        args.warmup = 3
        args.eval_eps = 3
        args.curriculum = 1
        print("[DRY-RUN] 1 seed, 3 warmup, 3 eval", flush=True)

    result = run(
        seeds=tuple(args.seeds),
        warmup_episodes=args.warmup,
        eval_episodes=args.eval_eps,
        steps_per_episode=args.steps,
        curriculum_episodes=args.curriculum,
        lr=args.lr,
        lr_enc=args.lr_enc,
        lr_rfm=args.lr_rfm,
        lr_wfm=args.lr_wfm,
        alpha_world=args.alpha_world,
        alpha_self=args.alpha_self,
        benefit_weight=args.benefit_weight,
        harm_weight=args.harm_weight,
    )

    if args.dry_run:
        print("\n[DRY-RUN] Smoke test complete. Not writing output file.", flush=True)
        print(f"Status: {result['status']}", flush=True)
        for k, v in result["metrics"].items():
            print(f"  {k}: {v}", flush=True)
        sys.exit(0)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"] = ts
    result["claim"] = CLAIM_IDS[0]
    result["verdict"] = result["status"]
    result["run_id"] = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = "ree_hybrid_guardrails_v1"

    out_dir = (
        Path(__file__).resolve().parents[2]
        / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{EXPERIMENT_TYPE}_{ts}.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"\nResult written to: {out_path}", flush=True)
    print(f"Status: {result['status']}", flush=True)
    for k, v in result["metrics"].items():
        print(f"  {k}: {v}", flush=True)
