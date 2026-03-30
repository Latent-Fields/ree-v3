#!/opt/local/bin/python3
"""
V3-EXQ-085o -- SD-015 Sorted-RF Proximity Encoder (Position-Invariant)

Claims: SD-015, SD-012, MECH-112

Supersedes: V3-EXQ-085l

=== ROOT CAUSE OF 085l C2 FAILURE ===

085l enc_loss=0.0011, prox_r2=0.908, goal_resource_r_enc=0.869.
BUT benefit_ratio=0.42x (ALL seeds below 1.0x -- goal guidance harmful):
  seed=42: present=0.130 vs absent=0.680  (0.19x)
  seed=7:  present=0.480 vs absent=0.730  (0.66x)
  seed=13: present=0.200 vs absent=0.520  (0.38x)

ROOT CAUSE: 15 of the 16 encoder dimensions are UNCONSTRAINED.
The proximity_head regression forces ONE linear direction to encode proximity.
The remaining 15 dimensions freely encode arbitrary features of resource_field_view --
including SPATIAL POSITION (which cell the resource occupies).

z_goal = EMA of enc.encode(rf) at contacts. Contact events encode both:
  (a) high proximity (1-dim proximity signal) -- position-invariant
  (b) specific cell pattern of the resource at contact time -- position-SPECIFIC

After resource respawn, the current rf has a different cell pattern.
cosine_sim(z_resource_curr, z_goal) is PENALISED by the 15-dim position mismatch,
overriding the 1-dim proximity signal. The agent is steered toward old resource
positions rather than current ones -- the same root cause as 085i, now in latent
space rather than raw rf space.

=== THE FIX: Sorted-RF Input ===

Sort resource_field_view values in descending order before passing to encoder:
  sorted_rf = rf.sort(descending=True).values  # [25] -> [25], position-FREE

Now contact at cell (2,1): sorted_rf = [1.0, 0.7, 0.5, 0.3, ...]
Contact at cell (7,8): sorted_rf = [1.0, 0.7, 0.5, 0.3, ...]
IDENTICAL. The encoder has NO spatial position to encode.

All 16 encoder dimensions must encode proximity-MAGNITUDE features (order statistics).
z_goal = EMA of enc.encode(sorted_rf) at contacts = "high-proximity magnitude profile."
cosine_sim(enc.encode(sorted_rf_curr), z_goal) is purely proximity-based.
No 15-dim position noise. Action selection finally has an unambiguous gradient.

RFM still operates on RAW rf (25-dim spatial grid) -- reliable dynamics preserved.
Action selection: argmax cosine_sim(enc.encode(sort(RFM(rf, a))), z_goal).
  RFM predicts rf_next in spatial space (position-aware, accurate).
  Sorting RF_pred removes position before goal comparison (position-free scoring).

=== PASS CRITERIA ===

C1: z_goal_norm_enc > 0.1        (sorted-rf goal is active)
C2: benefit_ratio >= 1.3x        (goal-guided nav beats random by 30%)
C3: goal_resource_r_enc > 0.2    (sorted-rf goal tracks resource proximity)
C4: prox_r2 > 0.3                (encoder's proximity regression has R^2 > 0.3)
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
from ree_core.goal import GoalState, GoalConfig
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_085o_sd015_sorted_rf_enc"
CLAIM_IDS = ["SD-015", "SD-012", "MECH-112"]

RESOURCE_OBS_DIM = 25  # resource_field_view: 5x5 proximity grid
ENCODER_DIM = 16       # z_resource embedding dimension


# ------------------------------------------------------------------ #
# ProximityEncoder (takes SORTED rf -- position-invariant)            #
# ------------------------------------------------------------------ #

class ProximityEncoder(nn.Module):
    """
    Learns z_resource = f(sorted_rf) via proximity regression (MSE).

    Input: sorted resource_field_view (descending order) -- position-FREE.
    Sorting removes spatial cell identity; all contacts at any position
    produce identical sorted_rf distributions.

    All 16 encoder dims encode proximity ORDER STATISTICS, not position.
    z_goal seeded from this encoder = position-invariant "high proximity" vector.
    cosine_sim(z_resource_current, z_goal) is purely proximity-magnitude-based.
    """

    def __init__(self, resource_obs_dim: int = 25, encoder_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(resource_obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, encoder_dim),
        )
        self.proximity_head = nn.Linear(encoder_dim, 1)

    def encode(self, sorted_rf: torch.Tensor) -> torch.Tensor:
        """sorted_rf: [B, 25] (sorted desc) -> z_resource: [B, ENCODER_DIM]"""
        return self.encoder(sorted_rf)

    def predict_proximity(self, sorted_rf: torch.Tensor) -> torch.Tensor:
        """sorted_rf: [B, 25] -> proximity_pred: [B, 1]"""
        return self.proximity_head(self.encoder(sorted_rf))


# ------------------------------------------------------------------ #
# ResourceForwardModel (raw rf space -- unchanged)                    #
# ------------------------------------------------------------------ #

class ResourceForwardModel(nn.Module):
    """
    Predicts resource_field_view_next from (resource_field_view_curr, action).
    Operates in raw 25-dim spatial space for accurate positional dynamics.
    RFM output is SORTED before passing to encoder for goal comparison.
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
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _sort_rf(rf: torch.Tensor) -> torch.Tensor:
    """Sort rf values descending. [B, 25] -> [B, 25]. Position-free."""
    return rf.sort(dim=-1, descending=True).values


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


def _rfm_sorted_enc_guided_action(
    rfm: ResourceForwardModel,
    enc: ProximityEncoder,
    goal_state_enc: GoalState,
    rf_curr: torch.Tensor,
    num_actions: int,
    device,
) -> int:
    """
    Hybrid action selection with sorted-rf position invariance:
      score(a) = cosine_sim(enc.encode(sort(RFM(rf_curr, a))), z_goal)

    RFM predicts next rf in spatial space (accurate positional dynamics).
    Sorting removes position before goal comparison -> invariant to respawn.
    Falls back to random if goal inactive.
    """
    if not goal_state_enc.is_active():
        return random.randint(0, num_actions - 1)
    with torch.no_grad():
        best_action = 0
        best_prox = -1.0
        for idx in range(num_actions):
            a = _action_to_onehot(idx, num_actions, device)
            rf_pred      = rfm(rf_curr, a)
            sorted_pred  = _sort_rf(rf_pred)
            z_pred       = enc.encode(sorted_pred)
            prox         = goal_state_enc.goal_proximity(z_pred).mean().item()
            if prox > best_prox:
                best_prox = prox
                best_action = idx
    return best_action


def _resource_proximity(env) -> float:
    if not env.resources:
        return 0.0
    ax, ay = env.agent_x, env.agent_y
    min_dist = min(abs(ax - int(r[0])) + abs(ay - int(r[1])) for r in env.resources)
    return 1.0 / (1.0 + min_dist)


def _pearson_r(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    if n < 4:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = (sum((x - mx) ** 2 for x in xs)) ** 0.5
    sy = (sum((y - my) ** 2 for y in ys)) ** 0.5
    if sx < 1e-9 or sy < 1e-9:
        return 0.0
    return num / (sx * sy)


def _r2_score(preds: List[float], targets: List[float]) -> float:
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
# Main run function                                                     #
# ------------------------------------------------------------------ #

def _run_single(
    seed: int,
    goal_present: bool,
    warmup_episodes: int,
    eval_episodes: int,
    steps_per_episode: int,
    curriculum_episodes: int,
    self_dim: int,
    world_dim: int,
    lr: float,
    lr_enc: float,
    lr_rfm: float,
    alpha_world: float,
    alpha_self: float,
    harm_scale: float,
    proximity_harm_scale: float,
    novelty_bonus_weight: float,
    drive_weight: float,
) -> Dict:
    torch.manual_seed(seed)
    random.seed(seed)

    cond_label = "GOAL_PRESENT" if goal_present else "GOAL_ABSENT"

    env = CausalGridWorldV2(
        seed=seed,
        size=10,
        num_hazards=3,
        num_resources=4,
        hazard_harm=harm_scale,
        env_drift_interval=5,
        env_drift_prob=0.1,
        proximity_harm_scale=proximity_harm_scale,
        proximity_benefit_scale=proximity_harm_scale * 0.6,
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
        novelty_bonus_weight=novelty_bonus_weight,
    )

    goal_config_world = GoalConfig(
        goal_dim=world_dim,
        alpha_goal=0.3,
        decay_goal=0.003,
        benefit_threshold=0.05,
        goal_weight=1.0,
        drive_weight=drive_weight,
        e1_goal_conditioned=True,
        z_goal_enabled=goal_present,
    )
    goal_state_world = GoalState(goal_config_world, device=torch.device("cpu"))

    goal_config_enc = GoalConfig(
        goal_dim=ENCODER_DIM,
        alpha_goal=0.5,
        decay_goal=0.003,
        benefit_threshold=0.05,
        goal_weight=1.0,
        drive_weight=drive_weight,
        e1_goal_conditioned=False,
        z_goal_enabled=goal_present,
    )
    goal_state_enc = GoalState(goal_config_enc, device=torch.device("cpu"))

    agent = REEAgent(config)
    enc = ProximityEncoder(resource_obs_dim=RESOURCE_OBS_DIM, encoder_dim=ENCODER_DIM)
    rfm = ResourceForwardModel(resource_dim=RESOURCE_OBS_DIM, action_dim=action_dim)

    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []
    prox_buf_rf:  List[torch.Tensor] = []  # sorted_rf
    prox_buf_tgt: List[float]        = []
    MAX_BUF = 4000

    standard_params = [p for n, p in agent.named_parameters()
                       if "harm_eval_head" not in n]
    harm_eval_params = list(agent.e3.harm_eval_head.parameters())
    optimizer     = optim.Adam(standard_params, lr=lr)
    harm_eval_opt = optim.Adam(harm_eval_params, lr=1e-4)
    enc_optimizer = optim.Adam(enc.parameters(), lr=lr_enc)
    rfm_optimizer = optim.Adam(rfm.parameters(), lr=lr_rfm)

    counts: Dict[str, int] = {
        "hazard_approach": 0, "env_caused_hazard": 0,
        "agent_caused_hazard": 0, "none": 0, "resource": 0,
        "benefit_approach": 0,
    }
    n_seedings_enc   = 0
    n_seedings_world = 0
    enc_train_losses: List[float] = []
    rfm_train_losses: List[float] = []
    prox_preds_recent: List[float] = []
    prox_tgts_recent:  List[float] = []

    agent.train()
    enc.train()
    rfm.train()

    prev_rf: Optional[torch.Tensor]        = None
    prev_action_oh: Optional[torch.Tensor] = None

    for ep in range(warmup_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        prev_rf        = None
        prev_action_oh = None

        if ep < curriculum_episodes:
            _place_resource_near_agent(env, max_dist=3)
            obs_dict = env._get_observation_dict()

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_harm  = obs_dict.get("harm_obs", None)
            rf_curr   = _ensure_2d(obs_dict["resource_field_view"].float())
            srf_curr  = _sort_rf(rf_curr)   # sorted -- position-free

            latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
            agent.clock.advance()
            z_world_curr = latent.z_world.detach()

            action_idx = random.randint(0, action_dim - 1)
            action_oh  = _action_to_onehot(action_idx, action_dim, agent.device)
            agent._last_action = action_oh

            prox_curr = _resource_proximity(env)

            _, harm_signal, done, info, obs_dict = env.step(action_oh)
            ttype = info.get("transition_type", "none")
            if ttype in counts:
                counts[ttype] += 1

            rf_next = _ensure_2d(obs_dict["resource_field_view"].float())

            # --- RFM training (raw spatial rf) ---
            if prev_rf is not None and prev_action_oh is not None:
                rf_pred  = rfm(prev_rf.detach(), prev_action_oh.detach())
                rfm_loss = F.mse_loss(rf_pred, rf_curr.detach())
                rfm_optimizer.zero_grad()
                rfm_loss.backward()
                torch.nn.utils.clip_grad_norm_(rfm.parameters(), 1.0)
                rfm_optimizer.step()
                rfm_train_losses.append(rfm_loss.item())

            prev_rf        = rf_curr.detach()
            prev_action_oh = action_oh.detach()

            # --- Encoder proximity regression (on SORTED rf) ---
            prox_buf_rf.append(srf_curr.detach())
            prox_buf_tgt.append(prox_curr)
            if len(prox_buf_rf) > MAX_BUF:
                prox_buf_rf  = prox_buf_rf[-MAX_BUF:]
                prox_buf_tgt = prox_buf_tgt[-MAX_BUF:]

            if len(prox_buf_rf) >= 16:
                k = min(32, len(prox_buf_rf))
                indices = random.sample(range(len(prox_buf_rf)), k)
                srf_batch = torch.cat([prox_buf_rf[i] for i in indices], dim=0)
                tgt_batch = torch.tensor(
                    [prox_buf_tgt[i] for i in indices],
                    dtype=torch.float32,
                ).unsqueeze(1)
                pred_prox = enc.predict_proximity(srf_batch)
                enc_loss  = F.mse_loss(pred_prox, tgt_batch)
                enc_optimizer.zero_grad()
                enc_loss.backward()
                torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
                enc_optimizer.step()
                enc_train_losses.append(enc_loss.item())
                with torch.no_grad():
                    prox_preds_recent.extend(pred_prox.squeeze(1).tolist())
                    prox_tgts_recent.extend(tgt_batch.squeeze(1).tolist())
                if len(prox_preds_recent) > 2000:
                    prox_preds_recent = prox_preds_recent[-2000:]
                    prox_tgts_recent  = prox_tgts_recent[-2000:]

            # --- Goal seeding: use sorted rf for position invariance ---
            if goal_present:
                if ttype == "resource":
                    goal_state_world.update(z_world_curr, benefit_exposure=1.0,
                                            drive_level=1.0)
                    with torch.no_grad():
                        z_resource_contact = enc.encode(srf_curr.detach())
                    goal_state_enc.update(z_resource_contact, benefit_exposure=1.0,
                                          drive_level=1.0)
                    n_seedings_world += 1
                    n_seedings_enc   += 1
                elif obs_body.shape[-1] > 11:
                    benefit_exposure = float(
                        obs_body[0, 11].item() if obs_body.dim() == 2
                        else obs_body[11].item()
                    )
                    energy = float(
                        obs_body[0, 3].item() if obs_body.dim() == 2
                        else obs_body[3].item()
                    )
                    drive_level = max(0.0, 1.0 - energy)
                    goal_state_world.update(z_world_curr, benefit_exposure,
                                            drive_level=drive_level)

            is_harm = float(harm_signal) < 0
            if is_harm:
                harm_buf_pos.append(z_world_curr)
                if len(harm_buf_pos) > 2000:
                    harm_buf_pos = harm_buf_pos[-2000:]
            else:
                harm_buf_neg.append(z_world_curr)
                if len(harm_buf_neg) > 2000:
                    harm_buf_neg = harm_buf_neg[-2000:]

            e1_loss    = agent.compute_prediction_loss()
            e2_loss    = agent.compute_e2_loss()
            total_loss = e1_loss + e2_loss
            if total_loss.requires_grad:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

            if len(harm_buf_pos) >= 4 and len(harm_buf_neg) >= 4:
                k_pos = min(16, len(harm_buf_pos))
                k_neg = min(16, len(harm_buf_neg))
                pos_idx = torch.randperm(len(harm_buf_pos))[:k_pos].tolist()
                neg_idx = torch.randperm(len(harm_buf_neg))[:k_neg].tolist()
                zw_pos  = torch.cat([harm_buf_pos[i] for i in pos_idx], dim=0)
                zw_neg  = torch.cat([harm_buf_neg[i] for i in neg_idx], dim=0)
                zw_b    = torch.cat([zw_pos, zw_neg], dim=0)
                target  = torch.cat([
                    torch.ones(k_pos,  1, device=agent.device),
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

            if done:
                break

        if (ep + 1) % 100 == 0 or ep == warmup_episodes - 1:
            enc_str = ""
            if enc_train_losses:
                recent  = enc_train_losses[-100:]
                enc_str = f" enc_loss={sum(recent)/max(1,len(recent)):.4f}"
            rfm_str = ""
            if rfm_train_losses:
                recent  = rfm_train_losses[-100:]
                rfm_str = f" rfm_loss={sum(recent)/max(1,len(recent)):.4f}"
            prox_r2        = _r2_score(prox_preds_recent, prox_tgts_recent)
            goal_norm_enc  = goal_state_enc.goal_norm() if goal_present else 0.0
            curriculum_tag = " [curriculum]" if ep < curriculum_episodes else ""
            print(
                f"  [train] seed={seed} cond={cond_label}"
                f" ep {ep+1}/{warmup_episodes}"
                f" resource={counts['resource']}"
                f" seedings_enc={n_seedings_enc}"
                f" z_goal_norm_enc={goal_norm_enc:.3f}"
                f" prox_r2={prox_r2:.3f}"
                f"{enc_str}{rfm_str}{curriculum_tag}",
                flush=True,
            )

    z_goal_norm_world_end = goal_state_world.goal_norm() if goal_present else 0.0
    z_goal_norm_enc_end   = goal_state_enc.goal_norm()   if goal_present else 0.0
    enc_final_loss = (
        float(sum(enc_train_losses[-500:]) / max(1, min(500, len(enc_train_losses))))
        if enc_train_losses else 0.0
    )
    rfm_final_loss = (
        float(sum(rfm_train_losses[-500:]) / max(1, min(500, len(rfm_train_losses))))
        if rfm_train_losses else 0.0
    )
    prox_r2_final = _r2_score(prox_preds_recent, prox_tgts_recent)

    # --- EVAL ---
    agent.eval()
    enc.eval()
    rfm.eval()

    benefit_per_ep: List[float] = []
    harm_buf_eval_pos: List[torch.Tensor] = []
    harm_buf_eval_neg: List[torch.Tensor] = []

    goal_prox_world_vals:  List[float] = []
    goal_prox_enc_vals:    List[float] = []
    resource_prox_vals:    List[float] = []
    harm_eval_vals:        List[float] = []
    goal_contrib_enc_vals: List[float] = []

    for _ in range(eval_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        ep_benefit = 0.0

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_harm  = obs_dict.get("harm_obs", None)
            rf_curr   = _ensure_2d(obs_dict["resource_field_view"].float())

            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
                agent.clock.advance()
                z_world_curr = latent.z_world.detach()

            if goal_present and goal_state_enc.is_active():
                action_idx = _rfm_sorted_enc_guided_action(
                    rfm, enc, goal_state_enc, rf_curr, action_dim, agent.device,
                )
            else:
                action_idx = random.randint(0, action_dim - 1)

            action_oh = _action_to_onehot(action_idx, action_dim, agent.device)
            agent._last_action = action_oh

            _, harm_signal, done, info, obs_dict = env.step(action_oh)
            ttype = info.get("transition_type", "none")

            if ttype == "resource":
                ep_benefit += 1.0
            if ttype == "benefit_approach" and obs_body.dim() == 2 and obs_body.shape[-1] > 11:
                ep_benefit += float(obs_body[0, 11].item()) * 0.1

            z_world_ev = z_world_curr
            if float(harm_signal) < 0:
                harm_buf_eval_pos.append(z_world_ev)
            else:
                harm_buf_eval_neg.append(z_world_ev)

            if goal_present:
                with torch.no_grad():
                    gp_world = (
                        goal_state_world.goal_proximity(z_world_ev).mean().item()
                        if goal_state_world.is_active() else 0.0
                    )
                    srf_c    = _sort_rf(rf_curr)
                    z_res_c  = enc.encode(srf_c)
                    gp_enc   = (
                        goal_state_enc.goal_proximity(z_res_c).mean().item()
                        if goal_state_enc.is_active() else 0.0
                    )
                    he = agent.e3.harm_eval(z_world_ev).mean().item()
                rp = _resource_proximity(env)
                goal_prox_world_vals.append(gp_world)
                goal_prox_enc_vals.append(gp_enc)
                resource_prox_vals.append(rp)
                harm_eval_vals.append(he)
                goal_contrib_enc_vals.append(goal_config_enc.goal_weight * gp_enc)

            if done:
                break

        benefit_per_ep.append(ep_benefit)

    avg_benefit = float(sum(benefit_per_ep) / max(1, len(benefit_per_ep)))

    calibration_gap = 0.0
    if len(harm_buf_eval_pos) >= 4 and len(harm_buf_eval_neg) >= 4:
        k = min(64, min(len(harm_buf_eval_pos), len(harm_buf_eval_neg)))
        zw_pos = torch.cat(harm_buf_eval_pos[-k:], dim=0)
        zw_neg = torch.cat(harm_buf_eval_neg[-k:], dim=0)
        with torch.no_grad():
            harm_pos = agent.e3.harm_eval(zw_pos).mean().item()
            harm_neg = agent.e3.harm_eval(zw_neg).mean().item()
        calibration_gap = harm_pos - harm_neg

    goal_resource_r_world = (
        _pearson_r(goal_prox_world_vals, resource_prox_vals) if goal_present else 0.0
    )
    goal_resource_r_enc = (
        _pearson_r(goal_prox_enc_vals, resource_prox_vals) if goal_present else 0.0
    )

    mean_goal_contrib_enc = (
        float(sum(goal_contrib_enc_vals) / max(1, len(goal_contrib_enc_vals)))
        if goal_present else 0.0
    )
    mean_harm_eval = (
        float(sum(harm_eval_vals) / max(1, len(harm_eval_vals)))
        if goal_present else 0.0
    )
    goal_vs_harm_ratio = (
        mean_goal_contrib_enc / max(1e-6, mean_harm_eval) if goal_present else 0.0
    )

    print(
        f"  [eval] seed={seed} cond={cond_label}"
        f" avg_benefit/ep={avg_benefit:.3f}"
        f" z_goal_norm_enc={z_goal_norm_enc_end:.3f}"
        f" enc_loss={enc_final_loss:.4f} rfm_loss={rfm_final_loss:.4f}"
        f" prox_r2={prox_r2_final:.3f}"
        f" cal_gap={calibration_gap:.4f}"
        f" goal_resource_r_enc={goal_resource_r_enc:.3f}"
        f" goal_vs_harm={goal_vs_harm_ratio:.3f}",
        flush=True,
    )

    return {
        "seed": seed,
        "condition": cond_label,
        "goal_present": goal_present,
        "avg_benefit_per_ep":    float(avg_benefit),
        "z_goal_norm_world_end": float(z_goal_norm_world_end),
        "z_goal_norm_enc_end":   float(z_goal_norm_enc_end),
        "n_seedings_enc":        int(n_seedings_enc),
        "calibration_gap":       float(calibration_gap),
        "goal_resource_r_world": float(goal_resource_r_world),
        "goal_resource_r_enc":   float(goal_resource_r_enc),
        "goal_vs_harm_ratio":    float(goal_vs_harm_ratio),
        "enc_final_loss":        float(enc_final_loss),
        "rfm_final_loss":        float(rfm_final_loss),
        "prox_r2":               float(prox_r2_final),
        "train_resource_events": int(counts["resource"]),
    }


def run(
    seeds: Tuple = (42, 7, 13),
    warmup_episodes: int = 800,
    eval_episodes: int = 100,
    steps_per_episode: int = 200,
    curriculum_episodes: int = 100,
    self_dim: int = 32,
    world_dim: int = 32,
    lr: float = 1e-3,
    lr_enc: float = 1e-3,
    lr_rfm: float = 5e-4,
    alpha_world: float = 0.9,
    alpha_self: float = 0.3,
    harm_scale: float = 0.02,
    proximity_harm_scale: float = 0.05,
    novelty_bonus_weight: float = 0.1,
    drive_weight: float = 2.0,
    **kwargs,
) -> dict:
    results_goal:    List[Dict] = []
    results_no_goal: List[Dict] = []

    for seed in seeds:
        for goal_present in [True, False]:
            label = "GOAL_PRESENT" if goal_present else "GOAL_ABSENT"
            print(
                f"\n[V3-EXQ-085o] {label} seed={seed}"
                f" warmup={warmup_episodes} eval={eval_episodes}"
                f" curriculum={curriculum_episodes}"
                f" drive_weight={drive_weight} alpha_world={alpha_world}"
                f" lr_enc={lr_enc} lr_rfm={lr_rfm}",
                flush=True,
            )
            r = _run_single(
                seed=seed,
                goal_present=goal_present,
                warmup_episodes=warmup_episodes,
                eval_episodes=eval_episodes,
                steps_per_episode=steps_per_episode,
                curriculum_episodes=curriculum_episodes,
                self_dim=self_dim,
                world_dim=world_dim,
                lr=lr,
                lr_enc=lr_enc,
                lr_rfm=lr_rfm,
                alpha_world=alpha_world,
                alpha_self=alpha_self,
                harm_scale=harm_scale,
                proximity_harm_scale=proximity_harm_scale,
                novelty_bonus_weight=novelty_bonus_weight,
                drive_weight=drive_weight,
            )
            if goal_present:
                results_goal.append(r)
            else:
                results_no_goal.append(r)

    def _avg(results: List[Dict], key: str) -> float:
        vals = [r[key] for r in results]
        return float(sum(vals) / max(1, len(vals)))

    z_goal_norm_world_avg  = _avg(results_goal, "z_goal_norm_world_end")
    z_goal_norm_enc_avg    = _avg(results_goal, "z_goal_norm_enc_end")
    benefit_goal_present   = _avg(results_goal,    "avg_benefit_per_ep")
    benefit_goal_absent    = _avg(results_no_goal, "avg_benefit_per_ep")
    cal_gap_goal_present   = _avg(results_goal, "calibration_gap")
    avg_goal_resource_r_w  = _avg(results_goal, "goal_resource_r_world")
    avg_goal_resource_r_e  = _avg(results_goal, "goal_resource_r_enc")
    avg_goal_vs_harm_ratio = _avg(results_goal, "goal_vs_harm_ratio")
    avg_enc_final_loss     = _avg(results_goal, "enc_final_loss")
    avg_rfm_final_loss     = _avg(results_goal, "rfm_final_loss")
    avg_prox_r2            = _avg(results_goal, "prox_r2")

    benefit_ratio = (
        benefit_goal_present / max(1e-6, benefit_goal_absent)
        if benefit_goal_absent > 1e-6 else 0.0
    )

    c1_pass = z_goal_norm_enc_avg > 0.1
    c2_pass = benefit_ratio >= 1.3
    c3_pass = avg_goal_resource_r_e > 0.2
    c4_pass = avg_prox_r2 > 0.3

    all_pass     = c1_pass and c2_pass and c3_pass and c4_pass
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass])
    status       = "PASS" if all_pass else "FAIL"

    if all_pass:
        decision = "retain_ree"
    elif c1_pass and c3_pass:
        decision = "hybridize"
    else:
        decision = "retire_ree_claim"

    mech124_flag_salience = avg_goal_vs_harm_ratio < 0.3

    print(f"\n[V3-EXQ-085o] Final results:", flush=True)
    print(
        f"  z_goal_norm_enc={z_goal_norm_enc_avg:.3f}",
        flush=True,
    )
    print(
        f"  benefit_goal_present={benefit_goal_present:.3f}"
        f"  benefit_goal_absent={benefit_goal_absent:.3f}"
        f"  ratio={benefit_ratio:.2f}x",
        flush=True,
    )
    print(
        f"  goal_resource_r_enc={avg_goal_resource_r_e:.3f}"
        f"  prox_r2={avg_prox_r2:.3f}",
        flush=True,
    )
    print(
        f"  enc_final_loss={avg_enc_final_loss:.4f}"
        f"  rfm_final_loss={avg_rfm_final_loss:.4f}",
        flush=True,
    )
    print(
        f"  goal_vs_harm={avg_goal_vs_harm_ratio:.3f}",
        flush=True,
    )
    print(
        f"  decision={decision}  status={status} ({criteria_met}/4)",
        flush=True,
    )

    failure_notes: List[str] = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: z_goal_norm_enc={z_goal_norm_enc_avg:.3f} <= 0.1"
        )
    if not c2_pass:
        if c3_pass:
            failure_notes.append(
                f"C2 FAIL: benefit_ratio={benefit_ratio:.2f}x < 1.3x BUT C3 PASS"
                f" (r={avg_goal_resource_r_e:.3f})."
                " Sorted-rf enc goal tracks proximity but nav gap persists."
                " If ratio > 0.9: proximity guidance nearly working, consider"
                " stronger action bias or more eval episodes."
                " If ratio < 0.5: likely residual position-specificity."
                " Next: direct predict_proximity action selection (no cosine_sim)."
            )
        else:
            failure_notes.append(
                f"C2 FAIL + C3 FAIL: full representation failure."
                f" prox_r2={avg_prox_r2:.3f}."
            )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: goal_resource_r_enc={avg_goal_resource_r_e:.3f} < 0.2."
        )
    if not c4_pass:
        failure_notes.append(
            f"C4 FAIL: prox_r2={avg_prox_r2:.3f} <= 0.3."
        )
    if mech124_flag_salience and c1_pass:
        failure_notes.append(
            f"MECH-124 V4 RISK: goal_vs_harm_ratio={avg_goal_vs_harm_ratio:.3f} < 0.3."
        )

    for note in failure_notes:
        print(f"  {note}", flush=True)

    per_goal_rows = "\n".join(
        f"  seed={r['seed']}:"
        f" benefit/ep={r['avg_benefit_per_ep']:.3f}"
        f" z_goal_norm_enc={r['z_goal_norm_enc_end']:.3f}"
        f" r_enc={r['goal_resource_r_enc']:.3f}"
        f" prox_r2={r['prox_r2']:.3f}"
        f" rfm_loss={r['rfm_final_loss']:.4f}"
        for r in results_goal
    )
    per_nogoal_rows = "\n".join(
        f"  seed={r['seed']}: benefit/ep={r['avg_benefit_per_ep']:.3f}"
        for r in results_no_goal
    )

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(
            f"- {n}" for n in failure_notes
        )

    summary_markdown = (
        f"# V3-EXQ-085o -- SD-015 Sorted-RF Proximity Encoder\n\n"
        f"**Status:** {status}\n"
        f"**Claims:** SD-015, SD-012, MECH-112\n"
        f"**Decision:** {decision}\n"
        f"**Seeds:** {list(seeds)}\n"
        f"**Supersedes:** V3-EXQ-085l\n\n"
        f"## Architecture\n\n"
        f"ProximityEncoder input: sorted_rf = rf.sort(descending=True) -- position-FREE.\n\n"
        f"**085l root cause:** proximity regression MSE trains 1-dim proximity_head"
        f" direction; remaining 15 dims of z_resource freely encode spatial position"
        f" (which cell the resource occupies). After respawn, 15-dim position mismatch"
        f" in cosine_sim overrides 1-dim proximity signal -- agent steered to old"
        f" resource location. All seeds below 1.0x (harmful guidance).\n\n"
        f"**Fix:** sorting rf removes all spatial position information. All contacts"
        f" at any cell produce identical sorted_rf profile ([1.0, 0.7, ...])."
        f" All 16 encoder dims encode proximity order statistics. z_goal is"
        f" position-invariant. cosine_sim purely proximity-based.\n\n"
        f"Action selection: cosine_sim(enc.encode(sort(RFM(rf,a))), z_goal).\n"
        f"RFM still operates on raw rf for accurate spatial dynamics.\n\n"
        f"**alpha_world:** {alpha_world}  (SD-008)\n"
        f"**SD-012:** drive_weight={drive_weight}, resource_respawn_on_consume=True\n"
        f"**lr_enc:** {lr_enc}  **lr_rfm:** {lr_rfm}\n"
        f"**enc_final_loss:** {avg_enc_final_loss:.4f}\n"
        f"**rfm_final_loss:** {avg_rfm_final_loss:.4f}\n"
        f"**prox_r2:** {avg_prox_r2:.3f}\n"
        f"**Warmup:** {warmup_episodes} eps"
        f" (curriculum={curriculum_episodes} eps)\n"
        f"**Eval:** {eval_episodes} eps"
        f" (GOAL_PRESENT: sorted-RF+enc hybrid; GOAL_ABSENT: random)\n\n"
        f"## Diagnostic Series Summary\n\n"
        f"| Variant | Enc input | goal_r | benefit_ratio | Note |\n"
        f"|---|---|---|---|---|\n"
        f"| 085i | raw rf (no enc) | 0.218 | 1.03x | spatially specific |\n"
        f"| 085j | BCE contact, raw rf | 0.819 | 0.034x | E2Resource degenerate |\n"
        f"| 085k | BCE contact, raw rf + RFM | 0.582 | 0.977x | binary enc near-random |\n"
        f"| 085l | prox regression, raw rf + RFM | 0.869 | 0.42x | 15-dim position noise |\n"
        f"| 085m | prox regression, sorted rf + RFM | {avg_goal_resource_r_e:.3f}"
        f" | {benefit_ratio:.2f}x | this run |\n\n"
        f"## Navigation Results\n\n"
        f"| Condition | benefit/ep | z_goal_norm_enc | r_enc | prox_r2 |\n"
        f"|---|---|---|---|---|\n"
        f"| GOAL_PRESENT | {benefit_goal_present:.3f} | {z_goal_norm_enc_avg:.3f}"
        f" | {avg_goal_resource_r_e:.3f} | {avg_prox_r2:.3f} |\n"
        f"| GOAL_ABSENT  | {benefit_goal_absent:.3f} | -- | -- | -- |\n\n"
        f"**Benefit ratio (goal/no-goal): {benefit_ratio:.2f}x**\n\n"
        f"## PASS Criteria\n\n"
        f"| Criterion | Result | Value |\n"
        f"|---|---|---|\n"
        f"| C1: z_goal_norm_enc > 0.1 | {'PASS' if c1_pass else 'FAIL'}"
        f" | {z_goal_norm_enc_avg:.3f} |\n"
        f"| C2: benefit ratio >= 1.3x | {'PASS' if c2_pass else 'FAIL'}"
        f" | {benefit_ratio:.2f}x |\n"
        f"| C3: goal_resource_r_enc > 0.2 | {'PASS' if c3_pass else 'FAIL'}"
        f" | {avg_goal_resource_r_e:.3f} |\n"
        f"| C4: prox_r2 > 0.3 | {'PASS' if c4_pass else 'FAIL'}"
        f" | {avg_prox_r2:.3f} |\n\n"
        f"Criteria met: {criteria_met}/4 -> **{status}**\n\n"
        f"## MECH-124 Diagnostics\n\n"
        f"goal_vs_harm_ratio: {avg_goal_vs_harm_ratio:.3f} (< 0.3 = V4 risk)\n\n"
        f"## Per-Seed\n\n"
        f"GOAL_PRESENT:\n{per_goal_rows}\n\n"
        f"GOAL_ABSENT:\n{per_nogoal_rows}\n"
        f"{failure_section}\n"
    )

    metrics = {
        "z_goal_norm_world_avg":        float(z_goal_norm_world_avg),
        "z_goal_norm_enc_avg":          float(z_goal_norm_enc_avg),
        "benefit_per_ep_goal_present":  float(benefit_goal_present),
        "benefit_per_ep_goal_absent":   float(benefit_goal_absent),
        "benefit_ratio":                float(benefit_ratio),
        "calibration_gap_goal_present": float(cal_gap_goal_present),
        "goal_resource_r_world":        float(avg_goal_resource_r_w),
        "goal_resource_r_enc":          float(avg_goal_resource_r_e),
        "goal_vs_harm_ratio":           float(avg_goal_vs_harm_ratio),
        "enc_final_loss":               float(avg_enc_final_loss),
        "rfm_final_loss":               float(avg_rfm_final_loss),
        "prox_r2":                      float(avg_prox_r2),
        "mech124_flag_salience":        float(mech124_flag_salience),
        "drive_weight":                 float(drive_weight),
        "n_seeds":                      float(len(seeds)),
        "alpha_world":                  float(alpha_world),
        "lr_enc":                       float(lr_enc),
        "lr_rfm":                       float(lr_rfm),
        "curriculum_episodes":          float(curriculum_episodes),
        "crit1_pass": 1.0 if c1_pass else 0.0,
        "crit2_pass": 1.0 if c2_pass else 0.0,
        "crit3_pass": 1.0 if c3_pass else 0.0,
        "crit4_pass": 1.0 if c4_pass else 0.0,
        "criteria_met": float(criteria_met),
    }

    return {
        "status": status,
        "metrics": metrics,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
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
    parser.add_argument("--warmup",          type=int,   default=800)
    parser.add_argument("--eval-eps",        type=int,   default=100)
    parser.add_argument("--steps",           type=int,   default=200)
    parser.add_argument("--curriculum",      type=int,   default=100)
    parser.add_argument("--alpha-world",     type=float, default=0.9)
    parser.add_argument("--alpha-self",      type=float, default=0.3)
    parser.add_argument("--harm-scale",      type=float, default=0.02)
    parser.add_argument("--proximity-scale", type=float, default=0.05)
    parser.add_argument("--novelty-weight",  type=float, default=0.1)
    parser.add_argument("--drive-weight",    type=float, default=2.0)
    parser.add_argument("--lr-enc",          type=float, default=1e-3)
    parser.add_argument("--lr-rfm",          type=float, default=5e-4)
    parser.add_argument("--dry-run",         action="store_true",
                        help="Run 1 seed, 5 warmup eps, 5 eval eps for smoke test")
    args = parser.parse_args()

    if args.dry_run:
        args.seeds    = [42]
        args.warmup   = 5
        args.eval_eps = 5
        args.curriculum = 2
        print("[DRY-RUN] 1 seed, 5 warmup, 5 eval", flush=True)

    result = run(
        seeds=tuple(args.seeds),
        warmup_episodes=args.warmup,
        eval_episodes=args.eval_eps,
        steps_per_episode=args.steps,
        curriculum_episodes=args.curriculum,
        alpha_world=args.alpha_world,
        alpha_self=args.alpha_self,
        harm_scale=args.harm_scale,
        proximity_harm_scale=args.proximity_scale,
        novelty_bonus_weight=args.novelty_weight,
        drive_weight=args.drive_weight,
        lr_enc=args.lr_enc,
        lr_rfm=args.lr_rfm,
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
