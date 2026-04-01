#!/opt/local/bin/python3
"""
V3-EXQ-185 -- Direct Proximity Argmax Navigation

Claims: MECH-112, SD-015

=== SCIENTIFIC QUESTION ===

Can the validated ProximityEncoder (prox_r2=0.91) and ResourceForwardModel
(rfm_loss=0.007) drive navigation when action selection is maximally simple?

Instead of cosine_sim to a stored z_goal snapshot (which is scene noise per
EXQ-179), DIRECTLY predict resource proximity for each candidate action and
pick the best.

=== KEY DIFFERENCE FROM 085l ===

085l: action = argmax cosine_sim(enc.encode(RFM(rf, a)), z_goal)
  -- FAILS because z_goal is scene-specific snapshot noise

EXQ-185: action = argmax enc.predict_proximity(RFM(rf, a))
  -- NO z_goal, NO cosine_sim, direct proximity maximization

=== DESIGN ===

Two conditions:
  GOAL_PRESENT: action = argmax_a enc.predict_proximity(RFM(rf_curr, a))
    Pick action whose predicted next-step resource_field_view has highest
    predicted proximity.

  GOAL_ABSENT: random action selection (baseline).

Warmup: 800 episodes random actions (curriculum: first 100 eps place
  resource near agent). Train REEAgent + enc + rfm + harm_eval.
Eval: 100 episodes per condition per seed.
Steps: 200 per episode.
Seeds: [42, 7, 13]

=== PASS CRITERIA ===

C1: benefit_ratio >= 1.3          (GOAL_PRESENT / GOAL_ABSENT)
C2: rfm_prox_discrimination > 0.05  (mean range of predicted proximity
    across 5 actions per step during eval -- confirms RFM predictions
    differ meaningfully across actions)
C3: prox_r2 > 0.3                (encoder proximity regression R^2)
C4: rfm_loss < 0.02              (RFM accurate enough)
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


EXPERIMENT_TYPE = "v3_exq_185_direct_prox_argmax"
CLAIM_IDS = ["MECH-112", "SD-015"]

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
    Reliable in raw 25-dim space: 085h/i loss=0.0078, 085k loss=0.0069.
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


def _direct_prox_argmax_action(
    rfm: ResourceForwardModel,
    enc: ProximityEncoder,
    rf_curr: torch.Tensor,
    num_actions: int,
    device,
) -> Tuple[int, List[float]]:
    """
    Direct proximity maximization: for each candidate action, predict the
    resource_field_view after taking that action via RFM, then predict
    resource proximity from the predicted rf via enc.predict_proximity.
    Pick the action with highest predicted proximity.

    Returns (best_action_idx, list_of_all_predicted_proximities).
    """
    with torch.no_grad():
        best_action = 0
        best_prox = -999.0
        action_proxs: List[float] = []
        for idx in range(num_actions):
            a = _action_to_onehot(idx, num_actions, device)
            rf_pred = rfm(rf_curr, a)
            prox = enc.predict_proximity(rf_pred).item()
            action_proxs.append(prox)
            if prox > best_prox:
                best_prox = prox
                best_action = idx
    return best_action, action_proxs


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
        novelty_bonus_weight=0.0,
    )

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
            curriculum_tag = " [curriculum]" if ep < curriculum_episodes else ""
            print(
                f"  [train] seed={seed} cond={cond_label}"
                f" ep {ep+1}/{warmup_episodes}"
                f" resource={counts['resource']}"
                f" prox_r2={prox_r2:.3f}"
                f"{enc_str}{rfm_str}{curriculum_tag}",
                flush=True,
            )

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
    harm_per_ep: List[float] = []
    rfm_prox_disc_vals: List[float] = []

    for eval_ep in range(eval_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        ep_benefit = 0.0
        ep_harm = 0.0

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_harm  = obs_dict.get("harm_obs", None)
            rf_curr   = _ensure_2d(obs_dict["resource_field_view"].float())

            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
                agent.clock.advance()

            # Action selection
            if goal_present:
                action_idx, action_proxs = _direct_prox_argmax_action(
                    rfm, enc, rf_curr, action_dim, agent.device,
                )
                # Track discrimination: max - min of predicted proximities
                if len(action_proxs) > 1:
                    disc = max(action_proxs) - min(action_proxs)
                    rfm_prox_disc_vals.append(disc)
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

            if float(harm_signal) < 0:
                ep_harm += abs(float(harm_signal))

            if done:
                break

        benefit_per_ep.append(ep_benefit)
        harm_per_ep.append(ep_harm)

    avg_benefit = float(sum(benefit_per_ep) / max(1, len(benefit_per_ep)))
    avg_harm = float(sum(harm_per_ep) / max(1, len(harm_per_ep)))
    avg_rfm_prox_disc = (
        float(sum(rfm_prox_disc_vals) / max(1, len(rfm_prox_disc_vals)))
        if rfm_prox_disc_vals else 0.0
    )

    print(
        f"  [eval] seed={seed} cond={cond_label}"
        f" avg_benefit/ep={avg_benefit:.3f}"
        f" avg_harm/ep={avg_harm:.3f}"
        f" enc_loss={enc_final_loss:.4f} rfm_loss={rfm_final_loss:.4f}"
        f" prox_r2={prox_r2_final:.3f}"
        f" rfm_prox_disc={avg_rfm_prox_disc:.4f}",
        flush=True,
    )

    return {
        "seed": seed,
        "condition": cond_label,
        "goal_present": goal_present,
        "avg_benefit_per_ep":    float(avg_benefit),
        "avg_harm_per_ep":       float(avg_harm),
        "enc_final_loss":        float(enc_final_loss),
        "rfm_final_loss":        float(rfm_final_loss),
        "prox_r2":               float(prox_r2_final),
        "rfm_prox_discrimination": float(avg_rfm_prox_disc),
        "train_resource_events": int(counts["resource"]),
    }


# ------------------------------------------------------------------ #
# Top-level run                                                        #
# ------------------------------------------------------------------ #

def run(
    seeds: Tuple = (42, 7, 13),
    warmup_episodes: int = 800,
    eval_episodes: int = 100,
    steps_per_episode: int = 200,
    curriculum_episodes: int = 100,
    self_dim: int = 16,
    world_dim: int = 32,
    lr: float = 1e-4,
    lr_enc: float = 1e-3,
    lr_rfm: float = 5e-4,
    alpha_world: float = 0.9,
    alpha_self: float = 0.3,
    harm_scale: float = 0.02,
    proximity_harm_scale: float = 0.006,
    **kwargs,
) -> dict:
    results_goal:    List[Dict] = []
    results_no_goal: List[Dict] = []

    for seed in seeds:
        for goal_present in [True, False]:
            label = "GOAL_PRESENT" if goal_present else "GOAL_ABSENT"
            print(
                f"\n[V3-EXQ-185] {label} seed={seed}"
                f" warmup={warmup_episodes} eval={eval_episodes}"
                f" curriculum={curriculum_episodes}"
                f" alpha_world={alpha_world}"
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
            )
            if goal_present:
                results_goal.append(r)
            else:
                results_no_goal.append(r)

    # ---- Aggregation ----
    def _avg(results: List[Dict], key: str) -> float:
        vals = [r[key] for r in results]
        return float(sum(vals) / max(1, len(vals)))

    benefit_goal_present   = _avg(results_goal,    "avg_benefit_per_ep")
    benefit_goal_absent    = _avg(results_no_goal, "avg_benefit_per_ep")
    harm_goal_present      = _avg(results_goal,    "avg_harm_per_ep")
    harm_goal_absent       = _avg(results_no_goal, "avg_harm_per_ep")
    avg_enc_final_loss     = _avg(results_goal, "enc_final_loss")
    avg_rfm_final_loss     = _avg(results_goal, "rfm_final_loss")
    avg_prox_r2            = _avg(results_goal, "prox_r2")
    avg_rfm_prox_disc      = _avg(results_goal, "rfm_prox_discrimination")

    benefit_ratio = (
        benefit_goal_present / max(1e-6, benefit_goal_absent)
        if benefit_goal_absent > 1e-6 else 0.0
    )

    # Pass criteria
    c1_pass = benefit_ratio >= 1.3
    c2_pass = avg_rfm_prox_disc > 0.05
    c3_pass = avg_prox_r2 > 0.3
    c4_pass = avg_rfm_final_loss < 0.02

    all_pass     = c1_pass and c2_pass and c3_pass and c4_pass
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass])
    status       = "PASS" if all_pass else "FAIL"

    if all_pass:
        decision = "retain_ree"
    elif c3_pass and c4_pass:
        decision = "hybridize"
    else:
        decision = "retire_ree_claim"

    print(f"\n[V3-EXQ-185] Final results:", flush=True)
    print(
        f"  benefit_goal_present={benefit_goal_present:.3f}"
        f"  benefit_goal_absent={benefit_goal_absent:.3f}"
        f"  ratio={benefit_ratio:.2f}x",
        flush=True,
    )
    print(
        f"  harm_goal_present={harm_goal_present:.3f}"
        f"  harm_goal_absent={harm_goal_absent:.3f}",
        flush=True,
    )
    print(
        f"  enc_final_loss={avg_enc_final_loss:.4f}"
        f"  rfm_final_loss={avg_rfm_final_loss:.4f}",
        flush=True,
    )
    print(
        f"  prox_r2={avg_prox_r2:.3f}"
        f"  rfm_prox_disc={avg_rfm_prox_disc:.4f}",
        flush=True,
    )
    print(
        f"  decision={decision}  status={status} ({criteria_met}/4)",
        flush=True,
    )

    failure_notes: List[str] = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: benefit_ratio={benefit_ratio:.2f}x < 1.3x."
            f" present={benefit_goal_present:.3f} vs absent={benefit_goal_absent:.3f}."
            " Direct proximity argmax does not produce 30% benefit improvement."
            " If RFM predictions barely differ across actions (C2 fail too),"
            " RFM is not learning action-specific predictions."
            " If C2 passes but C1 fails, proximity gradient exists but does"
            " not lead to resource collection -- env layout or harm dominance."
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: rfm_prox_discrimination={avg_rfm_prox_disc:.4f} <= 0.05."
            " RFM predicts nearly identical proximity for all actions."
            " The forward model does not distinguish action effects on proximity."
            " Possible cause: RFM trained on random actions sees too little"
            " variance, or rf space is too smooth to differentiate 1-step moves."
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: prox_r2={avg_prox_r2:.3f} <= 0.3."
            f" enc_loss={avg_enc_final_loss:.4f}."
            " Proximity regression not converged. Try: more warmup, higher lr_enc."
        )
    if not c4_pass:
        failure_notes.append(
            f"C4 FAIL: rfm_loss={avg_rfm_final_loss:.4f} >= 0.02."
            " ResourceForwardModel not accurate enough."
        )

    for note in failure_notes:
        print(f"  {note}", flush=True)

    per_goal_rows = "\n".join(
        f"  seed={r['seed']}:"
        f" benefit/ep={r['avg_benefit_per_ep']:.3f}"
        f" harm/ep={r['avg_harm_per_ep']:.3f}"
        f" prox_r2={r['prox_r2']:.3f}"
        f" rfm_loss={r['rfm_final_loss']:.4f}"
        f" rfm_prox_disc={r['rfm_prox_discrimination']:.4f}"
        for r in results_goal
    )
    per_nogoal_rows = "\n".join(
        f"  seed={r['seed']}:"
        f" benefit/ep={r['avg_benefit_per_ep']:.3f}"
        f" harm/ep={r['avg_harm_per_ep']:.3f}"
        for r in results_no_goal
    )

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(
            f"- {n}" for n in failure_notes
        )

    summary_markdown = (
        f"# V3-EXQ-185 -- Direct Proximity Argmax Navigation\n\n"
        f"**Status:** {status}\n"
        f"**Claims:** MECH-112, SD-015\n"
        f"**Decision:** {decision}\n"
        f"**Seeds:** {list(seeds)}\n\n"
        f"## Design\n\n"
        f"Direct proximity maximization: no z_goal, no cosine_sim.\n"
        f"action = argmax_a enc.predict_proximity(RFM(rf_curr, a))\n"
        f"Pick action whose predicted next-step resource_field_view has\n"
        f"highest predicted proximity.\n\n"
        f"Key difference from 085l: 085l uses cosine_sim to z_goal snapshot\n"
        f"(fails because z_goal is scene-specific noise per EXQ-179).\n"
        f"EXQ-185 bypasses z_goal entirely.\n\n"
        f"**alpha_world:** {alpha_world}  (SD-008)\n"
        f"**lr_enc:** {lr_enc}  **lr_rfm:** {lr_rfm}  **lr:** {lr}\n"
        f"**enc_final_loss (avg):** {avg_enc_final_loss:.4f}\n"
        f"**rfm_final_loss (avg):** {avg_rfm_final_loss:.4f}\n"
        f"**prox_r2 (avg):** {avg_prox_r2:.3f}\n"
        f"**rfm_prox_discrimination (avg):** {avg_rfm_prox_disc:.4f}\n"
        f"**Warmup:** {warmup_episodes} eps"
        f" (curriculum={curriculum_episodes} eps)\n"
        f"**Eval:** {eval_episodes} eps"
        f" (GOAL_PRESENT: direct prox argmax; GOAL_ABSENT: random)\n\n"
        f"## Navigation Results\n\n"
        f"| Condition | benefit/ep | harm/ep |\n"
        f"|---|---|---|\n"
        f"| GOAL_PRESENT | {benefit_goal_present:.3f} | {harm_goal_present:.3f} |\n"
        f"| GOAL_ABSENT  | {benefit_goal_absent:.3f} | {harm_goal_absent:.3f} |\n\n"
        f"**Benefit ratio (goal/no-goal): {benefit_ratio:.2f}x**\n\n"
        f"## PASS Criteria\n\n"
        f"| Criterion | Result | Value |\n"
        f"|---|---|---|\n"
        f"| C1: benefit_ratio >= 1.3x | {'PASS' if c1_pass else 'FAIL'}"
        f" | {benefit_ratio:.2f}x |\n"
        f"| C2: rfm_prox_disc > 0.05 | {'PASS' if c2_pass else 'FAIL'}"
        f" | {avg_rfm_prox_disc:.4f} |\n"
        f"| C3: prox_r2 > 0.3 | {'PASS' if c3_pass else 'FAIL'}"
        f" | {avg_prox_r2:.3f} |\n"
        f"| C4: rfm_loss < 0.02 | {'PASS' if c4_pass else 'FAIL'}"
        f" | {avg_rfm_final_loss:.4f} |\n\n"
        f"Criteria met: {criteria_met}/4 -- **{status}**\n\n"
        f"## Interpretation\n\n"
        f"If PASS: The ProximityEncoder + RFM pipeline can drive resource\n"
        f"navigation WITHOUT z_goal or cosine_sim. The bottleneck in 085l\n"
        f"was the z_goal mechanism, not the learned representations.\n\n"
        f"If C1 FAIL + C2 PASS: RFM produces differentiated predictions\n"
        f"but the proximity gradient does not translate to resource collection.\n"
        f"Possible cause: proximity gradient points toward resources but\n"
        f"agent encounters hazards en route, or resources respawn elsewhere.\n\n"
        f"If C2 FAIL: RFM does not differentiate actions -- forward model\n"
        f"is too coarse for 1-step proximity changes, or the 5x5 rf view\n"
        f"is too local to show action-specific differences.\n\n"
        f"## Per-Seed\n\n"
        f"GOAL_PRESENT:\n{per_goal_rows}\n\n"
        f"GOAL_ABSENT:\n{per_nogoal_rows}\n"
        f"{failure_section}\n"
    )

    metrics = {
        "benefit_per_ep_goal_present":  float(benefit_goal_present),
        "benefit_per_ep_goal_absent":   float(benefit_goal_absent),
        "benefit_ratio":                float(benefit_ratio),
        "harm_per_ep_goal_present":     float(harm_goal_present),
        "harm_per_ep_goal_absent":      float(harm_goal_absent),
        "enc_final_loss":               float(avg_enc_final_loss),
        "rfm_final_loss":               float(avg_rfm_final_loss),
        "prox_r2":                      float(avg_prox_r2),
        "rfm_prox_discrimination":      float(avg_rfm_prox_disc),
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

    per_seed_results = []
    for i, seed in enumerate(seeds):
        per_seed_results.append({
            "seed": seed,
            "goal_present": results_goal[i],
            "goal_absent": results_no_goal[i],
        })

    return {
        "status": status,
        "metrics": metrics,
        "per_seed_results": per_seed_results,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": (
            "supports" if all_pass
            else ("mixed" if criteria_met >= 2 else "weakens")
        ),
        "experiment_type": EXPERIMENT_TYPE,
        "fatal_error_count": 0,
    }


# ------------------------------------------------------------------ #
# Entry point                                                          #
# ------------------------------------------------------------------ #

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
    parser.add_argument("--proximity-scale", type=float, default=0.006)
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
