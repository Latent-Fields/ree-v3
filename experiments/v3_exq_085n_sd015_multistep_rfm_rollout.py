#!/opt/local/bin/python3
"""
V3-EXQ-085n -- SD-015 Multi-Step Greedy RFM Rollout

Claims: SD-015, SD-012, MECH-112

Supersedes: V3-EXQ-085l

=== ROOT CAUSE OF 085l C2 FAILURE ===

085l achieved:
  prox_r2=0.91 (proximity encoder works)
  goal_resource_r=0.87 (goal seeding works)
  goal_vs_harm_ratio=1.65 (goal salience OK)
  benefit_ratio=0.42x (FAIL -- nav benefit barely above random)

The representation is correct. The bottleneck is planning horizon.

1-step RFM lookahead: cosine_sim(enc.encode(RFM(rf, a)), z_goal)
This only informs action when a resource is within the 5x5 field of view (~30% of steps).
When the resource is outside the field, rf_pred looks almost identical for all 4 actions
=> all actions score equally => effectively random selection.

=== THE FIX: Multi-Step Greedy RFM Rollout ===

Replace 1-step lookahead with depth=5 greedy rollout.

For each candidate first action a0:
  - Simulate: rf1 = RFM(rf_curr, a0)
  - Track max cosine_sim at each depth step
  - For steps 2..depth: greedily pick best action to maximise cosine_sim at that step
  - Score(a0) = max cosine_sim across all depth steps

This extends the "visible horizon" of the action selector from 1 step to 5 steps,
increasing the fraction of steps where at least one action scores materially higher
than the others.

=== HYPOTHESIS (expected to FAIL) ===

Hypothesis: 5-step greedy rollout is sufficient to overcome the 1-step horizon limitation
and produce benefit_ratio >= 1.3x.

Expected result: FAIL. The RFM compound error at depth=5 (5 sequential 25-dim predictions,
each adding noise) may wash out the proximity gradient even faster than it extends it.
Resource is still outside view in ~70% of steps. The greedy intermediate selections do not
account for hazard avoidance. The encoder only represents current resource proximity, not
reachability accounting for grid obstacles or hazard zones.

This experiment "ties off" the planning horizon hypothesis. FAIL here confirms that
multi-step RFM rollout is insufficient, and that genuine hippocampal navigation (SD-004,
ARC-031) is required for goal-directed benefit acquisition at grid scale.

=== PASS CRITERIA ===

Same as 085l (retained for comparability):
C1: z_goal_norm_enc > 0.1        (proximity-encoded goal is active)
C2: benefit_ratio >= 1.3x        (goal-guided nav beats random by 30%)
C3: goal_resource_r_enc > 0.2    (encoded goal proximity tracks resource proximity)
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


EXPERIMENT_TYPE = "v3_exq_085n_sd015_multistep_rfm_rollout"
CLAIM_IDS = ["SD-015", "SD-012", "MECH-112"]

RESOURCE_OBS_DIM = 25  # resource_field_view: 5x5 proximity grid
ENCODER_DIM = 16       # z_resource embedding dimension
ROLLOUT_DEPTH = 5      # multi-step greedy lookahead depth


# ------------------------------------------------------------------ #
# ProximityEncoder                                                     #
# ------------------------------------------------------------------ #

class ProximityEncoder(nn.Module):
    """
    Learns z_resource = f(resource_field_view) via proximity regression (MSE).
    Training target: resource_proximity = 1 / (1 + manhattan_dist_to_nearest).
    Identical to 085l -- architecture not the bottleneck.
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
    Identical to 085l -- architecture not the bottleneck.
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


def _rfm_enc_guided_action_multistep(
    rfm: ResourceForwardModel,
    enc: ProximityEncoder,
    goal_state_enc: GoalState,
    rf_curr: torch.Tensor,
    num_actions: int,
    device,
    depth: int = ROLLOUT_DEPTH,
) -> int:
    """
    Multi-step greedy RFM rollout for action selection.

    For each candidate first action a0:
      1. Predict rf after a0: rf1 = RFM(rf_curr, a0)
      2. Track max cosine_sim(enc(rf), z_goal) across all depth steps
      3. At each intermediate step: greedily pick next action maximising cosine_sim
      4. Score(a0) = max cosine_sim along the depth-step trajectory

    Returns argmax Score(a0) over all first actions.

    Extends action selection horizon from 1 step to 5 steps, allowing the agent
    to "see" resources that are outside the 1-step field of view.
    """
    if not goal_state_enc.is_active():
        return random.randint(0, num_actions - 1)
    with torch.no_grad():
        best_a0 = 0
        best_score = -1.0
        for a0 in range(num_actions):
            rf = rfm(rf_curr, _action_to_onehot(a0, num_actions, device))
            z_pred = enc.encode(rf)
            max_sim = goal_state_enc.goal_proximity(z_pred).mean().item()
            for _ in range(depth - 1):
                # Greedy: pick best intermediate action
                best_inner = -1.0
                best_rf_next = rf
                for ai in range(num_actions):
                    rf_next = rfm(rf, _action_to_onehot(ai, num_actions, device))
                    sim = goal_state_enc.goal_proximity(
                        enc.encode(rf_next)
                    ).mean().item()
                    if sim > best_inner:
                        best_inner = sim
                        best_rf_next = rf_next
                rf = best_rf_next
                if best_inner > max_sim:
                    max_sim = best_inner
            if max_sim > best_score:
                best_score = max_sim
                best_a0 = a0
    return best_a0


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

    # z_world-seeded goal (baseline)
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

    # ProximityEncoder-seeded goal (SD-015)
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
    # Proximity regression replay buffer: (rf, proximity) pairs
    prox_buf_rf:   List[torch.Tensor] = []
    prox_buf_tgt:  List[float]        = []
    MAX_BUF = 4000

    standard_params = [p for n, p in agent.named_parameters()
                       if "harm_eval_head" not in n]
    harm_eval_params = list(agent.e3.harm_eval_head.parameters())
    optimizer      = optim.Adam(standard_params, lr=lr)
    harm_eval_opt  = optim.Adam(harm_eval_params, lr=1e-4)
    enc_optimizer  = optim.Adam(enc.parameters(), lr=lr_enc)
    rfm_optimizer  = optim.Adam(rfm.parameters(), lr=lr_rfm)

    counts: Dict[str, int] = {
        "hazard_approach": 0, "env_caused_hazard": 0,
        "agent_caused_hazard": 0, "none": 0, "resource": 0,
        "benefit_approach": 0,
    }
    n_seedings_enc   = 0
    n_seedings_world = 0
    enc_train_losses: List[float] = []
    rfm_train_losses: List[float] = []
    # Track recent proximity predictions vs targets for R^2
    prox_preds_recent:  List[float] = []
    prox_tgts_recent:   List[float] = []

    # --- WARMUP TRAINING ---
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

            latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
            agent.clock.advance()
            z_world_curr = latent.z_world.detach()

            action_idx = random.randint(0, action_dim - 1)
            action_oh  = _action_to_onehot(action_idx, action_dim, agent.device)
            agent._last_action = action_oh

            # Current resource proximity label (before step)
            prox_curr = _resource_proximity(env)

            _, harm_signal, done, info, obs_dict = env.step(action_oh)
            ttype = info.get("transition_type", "none")
            if ttype in counts:
                counts[ttype] += 1

            rf_next = _ensure_2d(obs_dict["resource_field_view"].float())

            # --- RFM training ---
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

            # --- Proximity regression training ---
            prox_buf_rf.append(rf_curr.detach())
            prox_buf_tgt.append(prox_curr)
            if len(prox_buf_rf) > MAX_BUF:
                prox_buf_rf  = prox_buf_rf[-MAX_BUF:]
                prox_buf_tgt = prox_buf_tgt[-MAX_BUF:]

            if len(prox_buf_rf) >= 16:
                k = min(32, len(prox_buf_rf))
                indices = random.sample(range(len(prox_buf_rf)), k)
                rf_batch  = torch.cat([prox_buf_rf[i] for i in indices], dim=0)
                tgt_batch = torch.tensor(
                    [prox_buf_tgt[i] for i in indices],
                    dtype=torch.float32,
                ).unsqueeze(1)
                pred_prox = enc.predict_proximity(rf_batch)
                enc_loss  = F.mse_loss(pred_prox, tgt_batch)
                enc_optimizer.zero_grad()
                enc_loss.backward()
                torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
                enc_optimizer.step()
                enc_train_losses.append(enc_loss.item())
                # Track for R^2
                with torch.no_grad():
                    preds = pred_prox.squeeze(1).tolist()
                    tgts  = tgt_batch.squeeze(1).tolist()
                prox_preds_recent.extend(preds)
                prox_tgts_recent.extend(tgts)
                if len(prox_preds_recent) > 2000:
                    prox_preds_recent = prox_preds_recent[-2000:]
                    prox_tgts_recent  = prox_tgts_recent[-2000:]

            # --- Goal seeding (GOAL_PRESENT only, contact-only for enc) ---
            if goal_present:
                if ttype == "resource":
                    goal_state_world.update(z_world_curr, benefit_exposure=1.0,
                                            drive_level=1.0)
                    with torch.no_grad():
                        z_resource_contact = enc.encode(rf_curr.detach())
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

            # --- Harm eval training ---
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
            prox_r2 = _r2_score(prox_preds_recent, prox_tgts_recent)
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

            # --- Multi-step greedy action selection ---
            if goal_present and goal_state_enc.is_active():
                action_idx = _rfm_enc_guided_action_multistep(
                    rfm, enc, goal_state_enc, rf_curr, action_dim, agent.device,
                    depth=ROLLOUT_DEPTH,
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
                    z_resource_curr = enc.encode(rf_curr)
                    gp_enc = (
                        goal_state_enc.goal_proximity(z_resource_curr).mean().item()
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
                f"\n[V3-EXQ-085n] {label} seed={seed}"
                f" warmup={warmup_episodes} eval={eval_episodes}"
                f" curriculum={curriculum_episodes}"
                f" drive_weight={drive_weight} alpha_world={alpha_world}"
                f" lr_enc={lr_enc} lr_rfm={lr_rfm}"
                f" rollout_depth={ROLLOUT_DEPTH}",
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

    print(f"\n[V3-EXQ-085n] Final results:", flush=True)
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
                " Multi-step RFM rollout (depth=5) insufficient for grid-scale navigation."
                " Conclusion: planning horizon not the bottleneck -- SD-004 hippocampal"
                " navigation required for goal-directed resource acquisition."
            )
        else:
            failure_notes.append(
                f"C2 FAIL: benefit_ratio={benefit_ratio:.2f}x"
                f" (present={benefit_goal_present:.3f} vs absent={benefit_goal_absent:.3f})"
            )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: goal_resource_r_enc={avg_goal_resource_r_e:.3f} < 0.2."
            f" prox_r2={avg_prox_r2:.3f}."
        )
    if not c4_pass:
        failure_notes.append(
            f"C4 FAIL: prox_r2={avg_prox_r2:.3f} <= 0.3."
            f" enc_loss={avg_enc_final_loss:.4f}."
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
        f"# V3-EXQ-085n -- SD-015 Multi-Step Greedy RFM Rollout\n\n"
        f"**Status:** {status}\n"
        f"**Claims:** SD-015, SD-012, MECH-112\n"
        f"**Decision:** {decision}\n"
        f"**Seeds:** {list(seeds)}\n"
        f"**Supersedes:** V3-EXQ-085l\n\n"
        f"## Architecture\n\n"
        f"ProximityEncoder (25->16, MSE regression) + ResourceForwardModel identical to 085l.\n"
        f"Action selection: multi-step greedy RFM rollout, depth={ROLLOUT_DEPTH}.\n\n"
        f"**Rationale:** 085l confirmed prox_r2=0.91, goal_resource_r=0.87 -- representation"
        f" is correct. benefit_ratio=0.42x because 1-step lookahead only informative when"
        f" resource is in 5x5 FOV (~30%% of steps). Depth=5 rollout should extend horizon"
        f" and allow action selection to see resources outside immediate FOV.\n\n"
        f"**Hypothesis (pre-registered as expected FAIL):** 5-step greedy rollout is"
        f" sufficient to overcome 1-step horizon limitation. Predicted FAIL because:"
        f" RFM compound error at depth=5 may wash out proximity gradient; greedy"
        f" intermediate selections ignore hazard zones; resource still outside extended"
        f" FOV in many steps. FAIL confirms SD-004 hippocampal navigation required.\n\n"
        f"**Rollout depth:** {ROLLOUT_DEPTH} steps\n"
        f"**Warmup:** {warmup_episodes} eps (curriculum={curriculum_episodes} eps)\n"
        f"**Eval:** {eval_episodes} eps\n"
        f"  (GOAL_PRESENT: depth={ROLLOUT_DEPTH} multistep RFM; GOAL_ABSENT: random)\n"
        f"**alpha_world:** {alpha_world}  (SD-008)\n"
        f"**SD-012:** drive_weight={drive_weight}, resource_respawn_on_consume=True\n"
        f"**lr_enc:** {lr_enc}  **lr_rfm:** {lr_rfm}\n"
        f"**enc_final_loss (avg):** {avg_enc_final_loss:.4f}\n"
        f"**rfm_final_loss (avg):** {avg_rfm_final_loss:.4f}\n"
        f"**prox_r2 (avg):** {avg_prox_r2:.3f}\n\n"
        f"## Diagnostic Series Summary\n\n"
        f"| Variant | Action selection | goal_r | benefit_ratio | Note |\n"
        f"|---|---|---|---|---|\n"
        f"| 085l | 1-step RFM | 0.87 | 0.42x | horizon too short |\n"
        f"| 085m | depth=5 greedy RFM | {avg_goal_resource_r_e:.3f}"
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
        "rollout_depth":                float(ROLLOUT_DEPTH),
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
