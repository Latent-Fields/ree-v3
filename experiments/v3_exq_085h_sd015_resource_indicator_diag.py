#!/opt/local/bin/python3
"""
V3-EXQ-085h -- SD-015 Resource Indicator Diagnostic

Claims: SD-015, SD-012, MECH-112

Supersedes: V3-EXQ-085g

=== DIAGNOSTIC CONTEXT ===

EXQ-085g FAIL analysis (2026-03-29):
  C1 PASS: z_goal_norm=0.399 (contact-gated seeding works)
  C3 PASS: cal_gap=0.218
  C2 FAIL: benefit_ratio=0.37x, goal_resource_r=0.066

Root cause (SD-015):
  z_goal seeded from z_world at resource contact. But z_world encodes the FULL
  scene (agent position, hazards, resource layout). Resources respawn randomly
  after consumption, so z_world at contact has near-zero predictive value for
  future resource locations. z_goal points at "that scene context" rather than
  "that kind of resource."

Counterintuitively, MORE seeding (085g: norm=0.399 vs 085f: norm=0.228) made
navigation WORSE (0.37x vs 0.28x). Higher norm just anchors z_goal more firmly
to scene-specific noise.

SD-015 hypothesis: A dedicated resource representation (resource_field_view --
the 5x5 proximity gradient field) encodes OBJECT-TYPE features invariant to
spatial layout. Seeding z_goal from resource_field_view should produce a goal
representation that correlates with resource proximity regardless of scene.

=== DIAGNOSTIC DESIGN ===

This experiment adds a ResourceForwardModel (RFM) alongside the existing REEAgent:

  RFM: resource_field_view_t + action -> resource_field_view_{t+1}

Trained during warmup on consecutive (rf, action, rf_next) triples.
Used in eval for E2-style lookahead in resource_field_view space.

TWO goal states maintained in GOAL_PRESENT condition:
  goal_state_world:    z_world-seeded (085g approach, goal_dim=world_dim=32)
  goal_state_resource: resource_field_view-seeded (SD-015, goal_dim=RESOURCE_DIM=25)

Action selection in eval: uses RFM + goal_state_resource (new approach).

KEY DIAGNOSTIC METRICS (both measured, compared):
  goal_resource_r_world:   r(cosine_sim(z_world_current, z_goal_world), resource_prox)
                           -- 085g approach, expected ~0.066
  goal_resource_r_rfm:     r(cosine_sim(rf_current, z_goal_resource), resource_prox)
                           -- new approach, DIAGNOSTIC CRITERION: > 0.2

=== PASS CRITERIA ===

C1: z_goal_resource_norm > 0.1  (RFM-seeded goal non-trivial)
C2: benefit_ratio >= 1.3x       (RFM-guided nav beats random by 30%)
C3: goal_resource_r_rfm > 0.2   (z_goal_resource points toward resources) [KEY DIAGNOSTIC]
C4: no fatal errors

Scientific interpretation:
  C3 PASS only: SD-015 representation hypothesis confirmed; E2_resource needed for C2
  C2+C3 PASS:   ResourceForwardModel sufficient; proceed to full SD-015 encoder
  C3 FAIL:      Deeper problem; resource_field_view itself may not encode useful info
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


EXPERIMENT_TYPE = "v3_exq_085h_sd015_resource_indicator_diag"
CLAIM_IDS = ["SD-015", "SD-012", "MECH-112"]

RESOURCE_DIM = 25  # resource_field_view: 5x5 proximity grid


# ------------------------------------------------------------------ #
# Resource Forward Model                                               #
# ------------------------------------------------------------------ #

class ResourceForwardModel(nn.Module):
    """
    Predicts resource_field_view_next from (resource_field_view_curr, action).

    Analog to E2.world_forward but operating in resource_field_view space.
    SD-015 diagnostic: enables E2-style lookahead to score goal proximity
    without requiring a learned ResourceEncoder or changes to REEAgent.

    Architecture: simple 2-hidden-layer MLP. The resource proximity field
    is a 5x5 grid; its dynamics are simple (field shifts as agent moves,
    resources respawn on consumption). A small MLP should learn this quickly.
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
        """rf: [1, RESOURCE_DIM], action: [1, action_dim] -> [1, RESOURCE_DIM]"""
        return self.net(torch.cat([rf, action], dim=-1))


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _ensure_2d(t: torch.Tensor) -> torch.Tensor:
    """Ensure tensor is [1, dim] (add batch dim if needed)."""
    if t.dim() == 1:
        return t.unsqueeze(0)
    return t


def _place_resource_near_agent(env, max_dist: int = 3) -> bool:
    """Place a resource within max_dist Manhattan steps of the agent (curriculum)."""
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


def _rfm_guided_action(
    rf_model: ResourceForwardModel,
    goal_state_res: GoalState,
    rf_curr: torch.Tensor,
    num_actions: int,
    device,
) -> int:
    """
    Pick action maximising goal_proximity(RFM(rf_curr, action)).

    Analog to _goal_guided_action in 085g but uses ResourceForwardModel
    and resource_field_view space instead of E2 and z_world space.
    Falls back to random if z_goal_resource is inactive (norm near zero).
    """
    if not goal_state_res.is_active():
        return random.randint(0, num_actions - 1)
    best_action = 0
    best_prox = -1.0
    with torch.no_grad():
        for idx in range(num_actions):
            a = _action_to_onehot(idx, num_actions, device)
            rf_pred = rf_model(rf_curr, a)
            prox = goal_state_res.goal_proximity(rf_pred).mean().item()
            if prox > best_prox:
                best_prox = prox
                best_action = idx
    return best_action


def _resource_proximity(env) -> float:
    """1 / (1 + manhattan_dist_to_nearest_resource). 0.0 if no resources."""
    if not env.resources:
        return 0.0
    ax, ay = env.agent_x, env.agent_y
    min_dist = min(abs(ax - int(r[0])) + abs(ay - int(r[1])) for r in env.resources)
    return 1.0 / (1.0 + min_dist)


def _pearson_r(xs: List[float], ys: List[float]) -> float:
    """Pearson r from two equal-length float lists. Returns 0.0 if degenerate."""
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

    # z_world-seeded goal state (085g approach -- comparison baseline)
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

    # resource_field_view-seeded goal state (SD-015 diagnostic -- new approach)
    goal_config_resource = GoalConfig(
        goal_dim=RESOURCE_DIM,
        alpha_goal=0.3,
        decay_goal=0.003,
        benefit_threshold=0.05,
        goal_weight=1.0,
        drive_weight=drive_weight,
        e1_goal_conditioned=False,  # RFM-based, not z_world-based
        z_goal_enabled=goal_present,
    )
    goal_state_resource = GoalState(goal_config_resource, device=torch.device("cpu"))

    agent = REEAgent(config)
    rf_model = ResourceForwardModel(resource_dim=RESOURCE_DIM, action_dim=action_dim)

    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []
    MAX_BUF = 2000

    standard_params = [
        p for n, p in agent.named_parameters()
        if "harm_eval_head" not in n
    ]
    harm_eval_params = list(agent.e3.harm_eval_head.parameters())
    optimizer = optim.Adam(standard_params, lr=lr)
    harm_eval_optimizer = optim.Adam(harm_eval_params, lr=1e-4)
    rf_optimizer = optim.Adam(rf_model.parameters(), lr=lr_rfm)

    counts: Dict[str, int] = {
        "hazard_approach": 0, "env_caused_hazard": 0,
        "agent_caused_hazard": 0, "none": 0, "resource": 0,
        "benefit_approach": 0,
    }
    n_resource_seedings_world    = 0
    n_resource_seedings_resource = 0
    rfm_train_losses: List[float] = []

    # --- WARMUP TRAINING ---
    agent.train()
    rf_model.train()

    prev_rf: Optional[torch.Tensor] = None
    prev_action_oh: Optional[torch.Tensor] = None

    for ep in range(warmup_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        prev_rf = None
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

            _, harm_signal, done, info, obs_dict = env.step(action_oh)
            ttype = info.get("transition_type", "none")
            if ttype in counts:
                counts[ttype] += 1

            rf_next = _ensure_2d(obs_dict["resource_field_view"].float())

            # --- RFM training ---
            if prev_rf is not None and prev_action_oh is not None:
                rf_pred = rf_model(prev_rf.detach(), prev_action_oh.detach())
                rfm_loss = F.mse_loss(rf_pred, rf_curr.detach())
                rf_optimizer.zero_grad()
                rfm_loss.backward()
                torch.nn.utils.clip_grad_norm_(rf_model.parameters(), 1.0)
                rf_optimizer.step()
                rfm_train_losses.append(rfm_loss.item())

            prev_rf = rf_curr.detach()
            prev_action_oh = action_oh.detach()

            # --- Goal seeding (GOAL_PRESENT only) ---
            if goal_present:
                if ttype == "resource":
                    # PRIMARY: contact-gated seeding (same logic as 085g)
                    # z_world-seeded baseline
                    goal_state_world.update(z_world_curr, benefit_exposure=1.0,
                                            drive_level=1.0)
                    # resource_field_view-seeded (SD-015 approach)
                    goal_state_resource.update(rf_curr.detach(), benefit_exposure=1.0,
                                               drive_level=1.0)
                    n_resource_seedings_world    += 1
                    n_resource_seedings_resource += 1
                elif obs_body.shape[-1] > 11:
                    # SECONDARY: proximity-based seeding (SD-012 drive modulation)
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
                    # resource_field_view secondary seeding
                    goal_state_resource.update(rf_curr.detach(), benefit_exposure,
                                               drive_level=drive_level)

            is_harm = float(harm_signal) < 0
            if is_harm:
                harm_buf_pos.append(z_world_curr)
                if len(harm_buf_pos) > MAX_BUF:
                    harm_buf_pos = harm_buf_pos[-MAX_BUF:]
            else:
                harm_buf_neg.append(z_world_curr)
                if len(harm_buf_neg) > MAX_BUF:
                    harm_buf_neg = harm_buf_neg[-MAX_BUF:]

            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
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
                zw_pos = torch.cat([harm_buf_pos[i] for i in pos_idx], dim=0)
                zw_neg = torch.cat([harm_buf_neg[i] for i in neg_idx], dim=0)
                zw_b   = torch.cat([zw_pos, zw_neg], dim=0)
                target = torch.cat([
                    torch.ones(k_pos,  1, device=agent.device),
                    torch.zeros(k_neg, 1, device=agent.device),
                ], dim=0)
                pred_harm = agent.e3.harm_eval(zw_b)
                harm_loss = F.mse_loss(pred_harm, target)
                if harm_loss.requires_grad:
                    harm_eval_optimizer.zero_grad()
                    harm_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        agent.e3.harm_eval_head.parameters(), 0.5,
                    )
                    harm_eval_optimizer.step()

            if done:
                break

        if (ep + 1) % 100 == 0 or ep == warmup_episodes - 1:
            rfm_loss_str = ""
            if rfm_train_losses:
                recent = rfm_train_losses[-100:]
                rfm_avg = sum(recent) / max(1, len(recent))
                rfm_loss_str = f" rfm_loss={rfm_avg:.4f}"
            goal_norm_world    = goal_state_world.goal_norm()
            goal_norm_resource = goal_state_resource.goal_norm() if goal_present else 0.0
            curriculum_tag = " [curriculum]" if ep < curriculum_episodes else ""
            print(
                f"  [train] seed={seed} cond={cond_label}"
                f" ep {ep+1}/{warmup_episodes}"
                f" resource={counts['resource']}"
                f" seedings_world={n_resource_seedings_world}"
                f" seedings_res={n_resource_seedings_resource}"
                f" z_goal_norm_world={goal_norm_world:.3f}"
                f" z_goal_norm_res={goal_norm_resource:.3f}"
                f"{rfm_loss_str}{curriculum_tag}",
                flush=True,
            )

    z_goal_norm_world_end    = goal_state_world.goal_norm() if goal_present else 0.0
    z_goal_norm_resource_end = goal_state_resource.goal_norm() if goal_present else 0.0
    rfm_final_loss = (
        float(sum(rfm_train_losses[-500:]) / max(1, min(500, len(rfm_train_losses))))
        if rfm_train_losses else 0.0
    )

    # --- EVAL ---
    agent.eval()
    rf_model.eval()

    benefit_per_ep: List[float] = []
    harm_buf_eval_pos: List[torch.Tensor] = []
    harm_buf_eval_neg: List[torch.Tensor] = []

    # Diagnostic accumulators (GOAL_PRESENT only)
    # For z_world-seeded goal (085g comparison)
    goal_prox_world_vals:    List[float] = []
    # For resource_field_view-seeded goal (SD-015)
    goal_prox_rfm_vals:      List[float] = []
    resource_prox_vals:      List[float] = []
    harm_eval_vals:          List[float] = []
    goal_contrib_world_vals: List[float] = []
    goal_contrib_rfm_vals:   List[float] = []

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

            # RFM-guided action selection in GOAL_PRESENT; random in GOAL_ABSENT
            if goal_present and goal_state_resource.is_active():
                action_idx = _rfm_guided_action(
                    rf_model, goal_state_resource, rf_curr,
                    action_dim, agent.device,
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

            # Diagnostic metrics (GOAL_PRESENT only)
            if goal_present:
                with torch.no_grad():
                    gp_world = (
                        goal_state_world.goal_proximity(z_world_ev).mean().item()
                        if goal_state_world.is_active() else 0.0
                    )
                    gp_rfm = (
                        goal_state_resource.goal_proximity(rf_curr).mean().item()
                        if goal_state_resource.is_active() else 0.0
                    )
                    he = agent.e3.harm_eval(z_world_ev).mean().item()
                rp = _resource_proximity(env)
                goal_prox_world_vals.append(gp_world)
                goal_prox_rfm_vals.append(gp_rfm)
                resource_prox_vals.append(rp)
                harm_eval_vals.append(he)
                goal_contrib_world_vals.append(goal_config_world.goal_weight * gp_world)
                goal_contrib_rfm_vals.append(goal_config_resource.goal_weight * gp_rfm)

            if done:
                break

        benefit_per_ep.append(ep_benefit)

    avg_benefit = float(sum(benefit_per_ep) / max(1, len(benefit_per_ep)))

    # Harm calibration check (C3)
    calibration_gap = 0.0
    if len(harm_buf_eval_pos) >= 4 and len(harm_buf_eval_neg) >= 4:
        k = min(64, min(len(harm_buf_eval_pos), len(harm_buf_eval_neg)))
        zw_pos = torch.cat(harm_buf_eval_pos[-k:], dim=0)
        zw_neg = torch.cat(harm_buf_eval_neg[-k:], dim=0)
        with torch.no_grad():
            harm_pos = agent.e3.harm_eval(zw_pos).mean().item()
            harm_neg = agent.e3.harm_eval(zw_neg).mean().item()
        calibration_gap = harm_pos - harm_neg

    # Diagnostic correlations
    goal_resource_r_world = (
        _pearson_r(goal_prox_world_vals, resource_prox_vals)
        if goal_present else 0.0
    )
    goal_resource_r_rfm = (
        _pearson_r(goal_prox_rfm_vals, resource_prox_vals)
        if goal_present else 0.0
    )

    # goal_vs_harm (MECH-124)
    mean_goal_contrib_rfm = (
        float(sum(goal_contrib_rfm_vals) / max(1, len(goal_contrib_rfm_vals)))
        if goal_present else 0.0
    )
    mean_harm_eval = (
        float(sum(harm_eval_vals) / max(1, len(harm_eval_vals)))
        if goal_present else 0.0
    )
    goal_vs_harm_ratio = (
        mean_goal_contrib_rfm / max(1e-6, mean_harm_eval)
        if goal_present else 0.0
    )

    print(
        f"  [eval] seed={seed} cond={cond_label}"
        f" avg_benefit/ep={avg_benefit:.3f}"
        f" z_goal_norm_world={z_goal_norm_world_end:.3f}"
        f" z_goal_norm_res={z_goal_norm_resource_end:.3f}"
        f" rfm_loss={rfm_final_loss:.4f}"
        f" cal_gap={calibration_gap:.4f}"
        f" goal_resource_r_world={goal_resource_r_world:.3f}"
        f" goal_resource_r_rfm={goal_resource_r_rfm:.3f}"
        f" goal_vs_harm={goal_vs_harm_ratio:.3f}",
        flush=True,
    )

    return {
        "seed": seed,
        "condition": cond_label,
        "goal_present": goal_present,
        "avg_benefit_per_ep": float(avg_benefit),
        "z_goal_norm_world_end": float(z_goal_norm_world_end),
        "z_goal_norm_resource_end": float(z_goal_norm_resource_end),
        "n_resource_seedings_world": int(n_resource_seedings_world),
        "n_resource_seedings_resource": int(n_resource_seedings_resource),
        "calibration_gap": float(calibration_gap),
        "goal_resource_r_world": float(goal_resource_r_world),
        "goal_resource_r_rfm": float(goal_resource_r_rfm),
        "goal_vs_harm_ratio": float(goal_vs_harm_ratio),
        "rfm_final_loss": float(rfm_final_loss),
        "train_resource_events": int(counts["resource"]),
    }


def run(
    seeds: Tuple = (42, 7, 13),
    warmup_episodes: int = 600,
    eval_episodes: int = 100,
    steps_per_episode: int = 200,
    curriculum_episodes: int = 100,
    self_dim: int = 32,
    world_dim: int = 32,
    lr: float = 1e-3,
    lr_rfm: float = 5e-4,
    alpha_world: float = 0.9,
    alpha_self: float = 0.3,
    harm_scale: float = 0.02,
    proximity_harm_scale: float = 0.05,
    novelty_bonus_weight: float = 0.1,
    drive_weight: float = 2.0,
    **kwargs,
) -> dict:
    """GOAL_PRESENT (RFM-guided eval) vs GOAL_ABSENT (random eval)."""
    results_goal:    List[Dict] = []
    results_no_goal: List[Dict] = []

    for seed in seeds:
        for goal_present in [True, False]:
            label = "GOAL_PRESENT" if goal_present else "GOAL_ABSENT"
            print(
                f"\n[V3-EXQ-085h] {label} seed={seed}"
                f" warmup={warmup_episodes} eval={eval_episodes}"
                f" curriculum={curriculum_episodes}"
                f" drive_weight={drive_weight} alpha_world={alpha_world}"
                f" lr_rfm={lr_rfm}",
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

    z_goal_norm_world_avg     = _avg(results_goal, "z_goal_norm_world_end")
    z_goal_norm_resource_avg  = _avg(results_goal, "z_goal_norm_resource_end")
    benefit_goal_present      = _avg(results_goal, "avg_benefit_per_ep")
    benefit_goal_absent       = _avg(results_no_goal, "avg_benefit_per_ep")
    cal_gap_goal_present      = _avg(results_goal, "calibration_gap")
    avg_goal_resource_r_world = _avg(results_goal, "goal_resource_r_world")
    avg_goal_resource_r_rfm   = _avg(results_goal, "goal_resource_r_rfm")
    avg_goal_vs_harm_ratio    = _avg(results_goal, "goal_vs_harm_ratio")
    avg_rfm_final_loss        = _avg(results_goal, "rfm_final_loss")

    benefit_ratio = (
        benefit_goal_present / max(1e-6, benefit_goal_absent)
        if benefit_goal_absent > 1e-6 else 0.0
    )

    c1_pass = z_goal_norm_resource_avg > 0.1
    c2_pass = benefit_ratio >= 1.3
    c3_pass = avg_goal_resource_r_rfm > 0.2   # KEY DIAGNOSTIC
    c4_pass = True

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass])
    status = "PASS" if all_pass else "FAIL"

    if all_pass:
        decision = "retain_ree"
    elif c1_pass and c3_pass:
        decision = "hybridize"   # SD-015 confirmed but full encoder needed for C2
    else:
        decision = "retire_ree_claim"

    mech124_flag_salience = avg_goal_vs_harm_ratio < 0.3

    print(f"\n[V3-EXQ-085h] Final results:", flush=True)
    print(
        f"  z_goal_norm_world={z_goal_norm_world_avg:.3f}"
        f"  z_goal_norm_resource={z_goal_norm_resource_avg:.3f}",
        flush=True,
    )
    print(
        f"  benefit_goal_present={benefit_goal_present:.3f}"
        f"  benefit_goal_absent={benefit_goal_absent:.3f}"
        f"  ratio={benefit_ratio:.2f}x",
        flush=True,
    )
    print(
        f"  goal_resource_r_world={avg_goal_resource_r_world:.3f}"
        f"  (085g baseline ~0.066)",
        flush=True,
    )
    print(
        f"  goal_resource_r_rfm={avg_goal_resource_r_rfm:.3f}"
        f"  (SD-015 diagnostic: need > 0.2)",
        flush=True,
    )
    print(
        f"  cal_gap={cal_gap_goal_present:.4f}"
        f"  goal_vs_harm={avg_goal_vs_harm_ratio:.3f}"
        f"  rfm_loss={avg_rfm_final_loss:.4f}",
        flush=True,
    )
    print(
        f"  decision={decision}  status={status} ({criteria_met}/4)",
        flush=True,
    )

    failure_notes: List[str] = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: z_goal_norm_resource={z_goal_norm_resource_avg:.3f} <= 0.1"
            " (resource_field_view-seeded goal not active -- check seedings > 0)"
        )
    if not c2_pass:
        if c3_pass:
            failure_notes.append(
                f"C2 FAIL: benefit_ratio={benefit_ratio:.2f}x < 1.3x BUT C3 PASS"
                f" (goal_resource_r_rfm={avg_goal_resource_r_rfm:.3f} > 0.2)."
                " SD-015 representation confirmed. RFM insufficient for navigation."
                " Next: full ResourceEncoder + E2_resource forward model (EXQ-085i)."
            )
        else:
            failure_notes.append(
                f"C2 FAIL: benefit_ratio={benefit_ratio:.2f}x < 1.3x"
                f" (goal_present={benefit_goal_present:.3f}"
                f" vs goal_absent={benefit_goal_absent:.3f})"
            )
    if not c3_pass:
        if avg_rfm_final_loss > 0.05:
            failure_notes.append(
                f"C3 FAIL: goal_resource_r_rfm={avg_goal_resource_r_rfm:.3f} < 0.2"
                f" AND rfm_loss={avg_rfm_final_loss:.4f} > 0.05."
                " RFM did not converge -- check warmup episodes (increase to 800+)"
                " or lr_rfm."
            )
        else:
            failure_notes.append(
                f"C3 FAIL: goal_resource_r_rfm={avg_goal_resource_r_rfm:.3f} < 0.2"
                f" despite rfm_loss={avg_rfm_final_loss:.4f}."
                " Deeper problem: resource_field_view itself may not be useful for goal."
                " Check: does CausalGridWorldV2 provide adequate resource_field_view"
                " variation? Are resources respawning as expected?"
            )
    if not c3_pass and avg_goal_resource_r_world > avg_goal_resource_r_rfm:
        failure_notes.append(
            "UNEXPECTED: z_world goal (085g approach) OUTPERFORMS resource_field_view."
            " SD-015 hypothesis may be wrong. Investigate env dynamics."
        )
    if mech124_flag_salience and c1_pass:
        failure_notes.append(
            f"MECH-124 V4 RISK: goal_vs_harm_ratio={avg_goal_vs_harm_ratio:.3f} < 0.3."
            " z_goal salience not competitive with harm salience."
        )

    for note in failure_notes:
        print(f"  {note}", flush=True)

    per_goal_rows = "\n".join(
        f"  seed={r['seed']}:"
        f" benefit/ep={r['avg_benefit_per_ep']:.3f}"
        f" z_goal_norm_res={r['z_goal_norm_resource_end']:.3f}"
        f" r_rfm={r['goal_resource_r_rfm']:.3f}"
        f" r_world={r['goal_resource_r_world']:.3f}"
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
        f"# V3-EXQ-085h -- SD-015 Resource Indicator Diagnostic\n\n"
        f"**Status:** {status}\n"
        f"**Claims:** SD-015, SD-012, MECH-112\n"
        f"**Decision:** {decision}\n"
        f"**Seeds:** {list(seeds)}\n"
        f"**Supersedes:** V3-EXQ-085g\n\n"
        f"## Diagnostic Design\n\n"
        f"Tests SD-015 hypothesis: resource_field_view (5x5 proximity grid) provides"
        f" a better goal representation than z_world at resource contact."
        f" ResourceForwardModel (RFM) trained during warmup to predict"
        f" resource_field_view_next given (resource_field_view_curr, action)."
        f" TWO goal states maintained: z_world-seeded (085g baseline) and"
        f" resource_field_view-seeded (SD-015). KEY DIAGNOSTIC: goal_resource_r_rfm > 0.2.\n\n"
        f"**alpha_world:** {alpha_world}  (SD-008)\n"
        f"**SD-012:** drive_weight={drive_weight}, resource_respawn_on_consume=True\n"
        f"**RFM lr:** {lr_rfm}\n"
        f"**rfm_final_loss (avg):** {avg_rfm_final_loss:.4f}\n"
        f"**Warmup:** {warmup_episodes} eps"
        f" (curriculum={curriculum_episodes} eps)\n"
        f"**Eval:** {eval_episodes} eps"
        f" (GOAL_PRESENT: RFM-lookahead goal guidance; GOAL_ABSENT: random)\n\n"
        f"## Key Diagnostic Comparison\n\n"
        f"| Representation | goal_resource_r | Note |\n"
        f"|---|---|---|\n"
        f"| z_world-seeded (085g) | {avg_goal_resource_r_world:.3f}"
        f" | expected ~0.066 |\n"
        f"| resource_field_view-seeded (SD-015) | {avg_goal_resource_r_rfm:.3f}"
        f" | DIAGNOSTIC: need > 0.2 |\n\n"
        f"z_goal_norm_world: {z_goal_norm_world_avg:.3f}  "
        f"z_goal_norm_resource: {z_goal_norm_resource_avg:.3f}\n\n"
        f"## Navigation Results\n\n"
        f"| Condition | benefit/ep | z_goal_norm_res | cal_gap | r_rfm | r_world |\n"
        f"|---|---|---|---|---|---|\n"
        f"| GOAL_PRESENT | {benefit_goal_present:.3f} | {z_goal_norm_resource_avg:.3f}"
        f" | {cal_gap_goal_present:.4f} | {avg_goal_resource_r_rfm:.3f}"
        f" | {avg_goal_resource_r_world:.3f} |\n"
        f"| GOAL_ABSENT  | {benefit_goal_absent:.3f} | -- | -- | -- | -- |\n\n"
        f"**Benefit ratio (goal/no-goal): {benefit_ratio:.2f}x**\n\n"
        f"## PASS Criteria\n\n"
        f"| Criterion | Result | Value |\n"
        f"|---|---|---|\n"
        f"| C1: z_goal_norm_resource > 0.1 | {'PASS' if c1_pass else 'FAIL'}"
        f" | {z_goal_norm_resource_avg:.3f} |\n"
        f"| C2: benefit ratio >= 1.3x | {'PASS' if c2_pass else 'FAIL'}"
        f" | {benefit_ratio:.2f}x |\n"
        f"| C3: goal_resource_r_rfm > 0.2 | {'PASS' if c3_pass else 'FAIL'}"
        f" | {avg_goal_resource_r_rfm:.3f} |\n"
        f"| C4: no fatal errors | {'PASS' if c4_pass else 'FAIL'} | -- |\n\n"
        f"Criteria met: {criteria_met}/4 -> **{status}**\n\n"
        f"## MECH-124 Diagnostics\n\n"
        f"goal_vs_harm_ratio: {avg_goal_vs_harm_ratio:.3f}"
        f" (< 0.3 = V4 risk)\n\n"
        f"## Per-Seed\n\n"
        f"GOAL_PRESENT:\n{per_goal_rows}\n\n"
        f"GOAL_ABSENT:\n{per_nogoal_rows}\n"
        f"{failure_section}\n"
    )

    metrics = {
        "z_goal_norm_world_avg":           float(z_goal_norm_world_avg),
        "z_goal_norm_resource_avg":        float(z_goal_norm_resource_avg),
        "benefit_per_ep_goal_present":     float(benefit_goal_present),
        "benefit_per_ep_goal_absent":      float(benefit_goal_absent),
        "benefit_ratio":                   float(benefit_ratio),
        "calibration_gap_goal_present":    float(cal_gap_goal_present),
        "goal_resource_r_world":           float(avg_goal_resource_r_world),
        "goal_resource_r_rfm":             float(avg_goal_resource_r_rfm),
        "goal_vs_harm_ratio":              float(avg_goal_vs_harm_ratio),
        "rfm_final_loss":                  float(avg_rfm_final_loss),
        "mech124_flag_salience":           float(mech124_flag_salience),
        "drive_weight":                    float(drive_weight),
        "n_seeds":                         float(len(seeds)),
        "alpha_world":                     float(alpha_world),
        "lr_rfm":                          float(lr_rfm),
        "curriculum_episodes":             float(curriculum_episodes),
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
    parser.add_argument("--warmup",          type=int,   default=600)
    parser.add_argument("--eval-eps",        type=int,   default=100)
    parser.add_argument("--steps",           type=int,   default=200)
    parser.add_argument("--curriculum",      type=int,   default=100)
    parser.add_argument("--alpha-world",     type=float, default=0.9)
    parser.add_argument("--alpha-self",      type=float, default=0.3)
    parser.add_argument("--harm-scale",      type=float, default=0.02)
    parser.add_argument("--proximity-scale", type=float, default=0.05)
    parser.add_argument("--novelty-weight",  type=float, default=0.1)
    parser.add_argument("--drive-weight",    type=float, default=2.0)
    parser.add_argument("--lr-rfm",          type=float, default=5e-4)
    parser.add_argument("--dry-run",         action="store_true",
                        help="Run 1 seed, 5 warmup eps, 5 eval eps for smoke test")
    args = parser.parse_args()

    if args.dry_run:
        args.seeds   = [42]
        args.warmup  = 5
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
