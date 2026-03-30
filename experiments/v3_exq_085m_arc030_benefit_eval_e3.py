#!/opt/local/bin/python3
"""
V3-EXQ-085m -- ARC-030 benefit_eval_head resource proximity + E3 selection

Claims: ARC-030, MECH-112, SD-015

=== CONTEXT: THE 085 SERIES ROOT CAUSE ===

The entire 085 series (085 through 085l) has been building and testing custom
action-selection modules (RFM, E2Resource, RFM+encoder hybrid, ProximityEncoder)
that BYPASS E3 entirely.

EXQ-179 tested E3 directly and found:
  H-A CONFIRMED: goal_tracking_r=-0.055 (z_world-seeded z_goal is scene noise)
  H-B: selection_bias=5e-7 (E3 not using goal -- but confounded by H-A failure)

The missing experiment: train E3's EXISTING benefit_eval_head on resource
proximity regression, enable benefit_eval_enabled=True, and let standard
E3.select() do the work.

=== WHY THIS IS THE RIGHT EXPERIMENT ===

1. resource_field_view[25] is already IN world_state[225:250].
   z_world is encoded from world_state. benefit_eval_head(z_world) has access
   to resource proximity information via the learned z_world representation.

2. benefit_eval_head is already in E3, already in score_trajectory().
   It just has never been trained on a useful target in any 085x experiment.

3. If trained to predict resource_proximity = max(resource_field_view), the
   benefit_eval_head becomes a "resource detector in z_world space".
   E3 then subtracts benefit_weight * benefit_score from trajectory cost,
   preferring trajectories that pass through resource-proximal z_world states.

This tests H-B directly WITHOUT the H-A confound:
  selection_bias_benefit = benefit_score(selected) - mean(benefit_score(others))
  If > 0.005: E3 IS selecting higher-benefit trajectories (H-B disconfirmed)
  If ~ 0: E3 cannot use benefit signal even when it's available

=== TRAINING DESIGN ===

benefit_eval_head trains on resource proximity regression:
  target = max(resource_field_view)  -- peak field value in 5x5 window
  loss = MSE(benefit_eval_head(z_world), target)

Every step provides a training sample. High proximity at contacts (~1.0),
low proximity far from resources (~0.0). Smooth gradient unlike binary BCE.

Separate optimizer for benefit_eval_head (lr=1e-4, same as harm_eval_head).

=== EXPERIMENTAL CONDITIONS ===

BENEFIT_ENABLED:  E3.select() with benefit_eval_enabled=True, benefit_weight=2.0
BENEFIT_DISABLED: E3.select() with benefit_eval_enabled=False (harm+residue only)

Both conditions train benefit_eval_head during warmup.
Difference is only whether benefit scores affect trajectory selection during eval.
This cleanly measures the causal contribution of benefit evaluation to navigation.

No goal_state / z_goal attractor in this experiment.
Pure ARC-030 test: can trained benefit_eval_head guide E3?

=== PASS CRITERIA ===

C1: benefit_eval_r2 > 0.3     (benefit_eval_head learned resource proximity from z_world)
C2: benefit_ratio >= 1.3x     (benefit-guided E3 nav beats harm-only E3 by 30%)
C3: selection_bias_benefit > 0.005 (E3 preferentially picks higher-benefit trajectories)
C4: calibration_gap > 0.0    (benefit_eval_head assigns higher scores near resources)

Scientific interpretation:
  All PASS:    ARC-030 confirmed; E3 benefit_eval_head drives resource navigation.
  C1 FAIL:     z_world doesn't encode resource proximity (benefit_eval_head can't learn it)
  C1 PASS, C3 FAIL:  E3 not using benefit scores (H-B confirmed independently of H-A)
  C1+C3 PASS, C2 FAIL: E3 selects "better" trajectories but planning horizon or
                        harm-benefit balance prevents closing distance (tuning issue)
"""

import sys
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_085m_arc030_benefit_eval_e3"
CLAIM_IDS = ["ARC-030", "MECH-112", "SD-015"]


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _resource_proximity(env) -> float:
    """1 / (1 + manhattan_dist_to_nearest_resource). 0.0 if no resources."""
    if not env.resources:
        return 0.0
    ax, ay = env.agent_x, env.agent_y
    min_dist = min(abs(ax - int(r[0])) + abs(ay - int(r[1])) for r in env.resources)
    return 1.0 / (1.0 + min_dist)


def _pearson_r(xs: List[float], ys: List[float]) -> float:
    """Pearson r. Returns 0.0 if degenerate."""
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


def _r_squared(pred: List[float], actual: List[float]) -> float:
    """R^2 = 1 - SS_res / SS_tot."""
    n = len(pred)
    if n < 4:
        return 0.0
    mean_a = sum(actual) / n
    ss_tot = sum((a - mean_a) ** 2 for a in actual)
    ss_res = sum((p - a) ** 2 for p, a in zip(pred, actual))
    if ss_tot < 1e-9:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _place_resource_near_agent(env, max_dist: int = 3) -> bool:
    """Curriculum: place resource within max_dist Manhattan steps of agent."""
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


def _ensure_2d(t: torch.Tensor) -> torch.Tensor:
    if t.dim() == 1:
        return t.unsqueeze(0)
    return t


# ------------------------------------------------------------------ #
# Main run function                                                     #
# ------------------------------------------------------------------ #

def _run_single(
    seed: int,
    benefit_enabled: bool,
    warmup_episodes: int,
    eval_episodes: int,
    steps_per_episode: int,
    curriculum_episodes: int,
    self_dim: int,
    world_dim: int,
    lr: float,
    alpha_world: float,
    alpha_self: float,
    harm_scale: float,
    proximity_harm_scale: float,
    novelty_bonus_weight: float,
    benefit_weight: float,
    temperature: float,
) -> Dict:
    torch.manual_seed(seed)
    random.seed(seed)

    cond_label = "BENEFIT_ENABLED" if benefit_enabled else "BENEFIT_DISABLED"

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

    # benefit_eval_head trained during warmup regardless of condition.
    # benefit_eval_enabled toggles whether E3.score_trajectory() uses it during eval.
    # During warmup, always disabled (random actions, consistent across conditions).
    config.e3.benefit_eval_enabled = False   # off during warmup
    config.e3.benefit_weight = benefit_weight

    agent = REEAgent(config)

    standard_params = [
        p for n, p in agent.named_parameters()
        if "harm_eval_head" not in n and "benefit_eval_head" not in n
    ]
    harm_eval_params = list(agent.e3.harm_eval_head.parameters())
    benefit_eval_params = list(agent.e3.benefit_eval_head.parameters())
    optimizer = optim.Adam(standard_params, lr=lr)
    harm_eval_optimizer = optim.Adam(harm_eval_params, lr=1e-4)
    benefit_eval_optimizer = optim.Adam(benefit_eval_params, lr=1e-4)

    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []
    MAX_BUF = 2000

    # Running buffers for benefit_eval R^2 tracking (warmup)
    benefit_pred_vals:   List[float] = []
    benefit_target_vals: List[float] = []

    counts: Dict[str, int] = {
        "hazard_approach": 0, "env_caused_hazard": 0,
        "agent_caused_hazard": 0, "none": 0, "resource": 0,
        "benefit_approach": 0,
    }
    benefit_train_losses: List[float] = []

    # --- WARMUP TRAINING ---
    agent.train()

    for ep in range(warmup_episodes):
        _, obs_dict = env.reset()
        agent.reset()

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

            # --- benefit_eval_head training: resource proximity regression ---
            # Target: peak resource field value in 5x5 window (0=no resource, 1=contact)
            resource_prox_target = float(rf_curr.max().item())
            prox_t = torch.tensor(
                [[resource_prox_target]], dtype=torch.float32, device=agent.device
            )
            with torch.no_grad():
                benefit_pred = agent.e3.benefit_eval_head(z_world_curr)
            benefit_train_losses.append(
                float(F.mse_loss(benefit_pred.detach(), prox_t.detach()).item())
            )
            # Full forward + backward for training
            benefit_pred_train = agent.e3.benefit_eval_head(z_world_curr)
            b_loss = F.mse_loss(benefit_pred_train, prox_t.detach())
            if b_loss.requires_grad:
                benefit_eval_optimizer.zero_grad()
                b_loss.backward()
                torch.nn.utils.clip_grad_norm_(benefit_eval_params, 0.5)
                benefit_eval_optimizer.step()

            # Track R^2 for final reporting (last 1000 steps)
            benefit_pred_vals.append(float(benefit_pred.item()))
            benefit_target_vals.append(resource_prox_target)
            if len(benefit_pred_vals) > 2000:
                benefit_pred_vals  = benefit_pred_vals[-2000:]
                benefit_target_vals = benefit_target_vals[-2000:]

            # Record that a benefit sample was seen (for warmup gate)
            agent.e3.record_benefit_sample(1)

            # --- harm_eval_head training ---
            is_harm = float(harm_signal) < 0
            if is_harm:
                harm_buf_pos.append(z_world_curr)
                if len(harm_buf_pos) > MAX_BUF:
                    harm_buf_pos = harm_buf_pos[-MAX_BUF:]
            else:
                harm_buf_neg.append(z_world_curr)
                if len(harm_buf_neg) > MAX_BUF:
                    harm_buf_neg = harm_buf_neg[-MAX_BUF:]

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
                    torch.nn.utils.clip_grad_norm_(harm_eval_params, 0.5)
                    harm_eval_optimizer.step()

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

        if (ep + 1) % 50 == 0 or ep == warmup_episodes - 1:
            r2 = _r_squared(benefit_pred_vals[-500:], benefit_target_vals[-500:])
            recent_b_loss = benefit_train_losses[-200:]
            b_loss_avg = sum(recent_b_loss) / max(1, len(recent_b_loss))
            print(
                f"  [train] seed={seed} cond={cond_label}"
                f" ep {ep+1}/{warmup_episodes}"
                f" resource={counts['resource']}"
                f" benefit_r2={r2:.3f}"
                f" benefit_loss={b_loss_avg:.4f}",
                flush=True,
            )

    # Final benefit_eval R^2 over last 1000 training samples
    benefit_eval_r2_final = _r_squared(
        benefit_pred_vals[-1000:], benefit_target_vals[-1000:]
    )
    benefit_loss_final = (
        float(sum(benefit_train_losses[-500:]) / max(1, min(500, len(benefit_train_losses))))
        if benefit_train_losses else 0.0
    )

    # Enable benefit scoring in E3 for BENEFIT_ENABLED condition
    config.e3.benefit_eval_enabled = benefit_enabled

    # --- EVAL ---
    agent.eval()

    benefit_per_ep: List[float] = []

    # Selection bias measurement (H-B analog for benefit scores)
    selection_bias_vals:      List[float] = []
    benefit_score_rank_vals:  List[float] = []

    # benefit_eval calibration: high near resources, low far
    benefit_near_resource_vals: List[float] = []
    benefit_far_resource_vals:  List[float] = []

    # Harm calibration
    harm_buf_eval_pos: List[torch.Tensor] = []
    harm_buf_eval_neg: List[torch.Tensor] = []

    # Tracking: benefit_eval score vs actual resource proximity
    benefit_prox_pred_vals:   List[float] = []
    benefit_prox_actual_vals: List[float] = []

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
            ticks = agent.clock.advance()
            with torch.no_grad():
                z_world_curr = latent.z_world.detach()

            # E3 action selection (with or without benefit scoring)
            if ticks.get("e3_tick", True):
                with torch.no_grad():
                    e1_prior = agent._e1_tick(latent)
                    candidates = agent._e3_tick(latent, e1_prior)

                    # Compute benefit score for every candidate (for selection bias)
                    all_benefit_scores = [
                        agent.e3.compute_benefit_score(t).mean().item()
                        for t in candidates
                    ]

                    result = agent.e3.select(
                        candidates, temperature=temperature,
                    )
                action_oh = result.selected_action
                agent._last_action = action_oh

                # Selection bias: does E3 pick trajectories with higher benefit?
                if len(all_benefit_scores) > 1:
                    bs_sel = all_benefit_scores[result.selected_index]
                    others = [
                        s for i, s in enumerate(all_benefit_scores)
                        if i != result.selected_index
                    ]
                    selection_bias_vals.append(bs_sel - (sum(others) / len(others)))
                    n_beaten = sum(1 for s in all_benefit_scores if bs_sel > s)
                    benefit_score_rank_vals.append(
                        n_beaten / max(1, len(all_benefit_scores) - 1)
                    )
            else:
                # Hold action between E3 ticks (MECH-057a)
                if agent._last_action is None:
                    action_oh = _action_to_onehot(
                        random.randint(0, action_dim - 1), action_dim, agent.device
                    )
                    agent._last_action = action_oh
                else:
                    action_oh = agent._last_action

            _, harm_signal, done, info, obs_dict = env.step(action_oh)
            ttype = info.get("transition_type", "none")

            if ttype == "resource":
                ep_benefit += 1.0
            if ttype == "benefit_approach" and obs_body.dim() == 2 and obs_body.shape[-1] > 11:
                ep_benefit += float(obs_body[0, 11].item()) * 0.1

            # Calibration: benefit_eval near vs far resource
            rp = _resource_proximity(env)
            with torch.no_grad():
                b_pred = agent.e3.benefit_eval_head(z_world_curr).mean().item()
            benefit_prox_pred_vals.append(b_pred)
            benefit_prox_actual_vals.append(float(rf_curr.max().item()))

            if rp > 0.33:  # within 2 cells
                benefit_near_resource_vals.append(b_pred)
            elif rp < 0.1:  # far away
                benefit_far_resource_vals.append(b_pred)

            if float(harm_signal) < 0:
                harm_buf_eval_pos.append(z_world_curr)
            else:
                harm_buf_eval_neg.append(z_world_curr)

            if done:
                break

        benefit_per_ep.append(ep_benefit)

    avg_benefit = float(sum(benefit_per_ep) / max(1, len(benefit_per_ep)))

    # benefit_eval calibration gap (high near resources, low far)
    calibration_gap_benefit = 0.0
    if benefit_near_resource_vals and benefit_far_resource_vals:
        calibration_gap_benefit = (
            float(sum(benefit_near_resource_vals) / len(benefit_near_resource_vals))
            - float(sum(benefit_far_resource_vals) / len(benefit_far_resource_vals))
        )

    # benefit_eval R^2 during eval (tracking resource proximity)
    benefit_eval_r2_eval = _r_squared(benefit_prox_pred_vals, benefit_prox_actual_vals)

    # Selection bias metrics
    mean_selection_bias = (
        float(sum(selection_bias_vals) / max(1, len(selection_bias_vals)))
        if selection_bias_vals else 0.0
    )
    mean_rank_pct = (
        float(sum(benefit_score_rank_vals) / max(1, len(benefit_score_rank_vals)))
        if benefit_score_rank_vals else 0.0
    )

    # harm calibration gap
    harm_calibration_gap = 0.0
    if len(harm_buf_eval_pos) >= 4 and len(harm_buf_eval_neg) >= 4:
        k = min(64, min(len(harm_buf_eval_pos), len(harm_buf_eval_neg)))
        zw_pos = torch.cat(harm_buf_eval_pos[-k:], dim=0)
        zw_neg = torch.cat(harm_buf_eval_neg[-k:], dim=0)
        with torch.no_grad():
            harm_pos = agent.e3.harm_eval(zw_pos).mean().item()
            harm_neg = agent.e3.harm_eval(zw_neg).mean().item()
        harm_calibration_gap = harm_pos - harm_neg

    # benefit_eval_head vs harm_eval_head salience ratio (MECH-124 analog)
    mean_benefit_pred = (
        float(sum(benefit_prox_pred_vals) / max(1, len(benefit_prox_pred_vals)))
        if benefit_prox_pred_vals else 0.0
    )
    mean_harm_pred = float(
        agent.e3.harm_eval(
            torch.stack(harm_buf_eval_neg[-50:]) if harm_buf_eval_neg else
            torch.zeros(1, world_dim)
        ).mean().item()
    )
    benefit_vs_harm_ratio = mean_benefit_pred / max(1e-6, mean_harm_pred)

    print(
        f"  [eval] seed={seed} cond={cond_label}"
        f" avg_benefit/ep={avg_benefit:.3f}"
        f" benefit_r2_eval={benefit_eval_r2_eval:.3f}"
        f" cal_gap={calibration_gap_benefit:.4f}"
        f" sel_bias={mean_selection_bias:.5f}"
        f" rank_pct={mean_rank_pct:.3f}"
        f" benefit_vs_harm={benefit_vs_harm_ratio:.3f}",
        flush=True,
    )

    return {
        "seed": seed,
        "condition": cond_label,
        "benefit_enabled": benefit_enabled,
        "avg_benefit_per_ep": float(avg_benefit),
        "benefit_eval_r2_train": float(benefit_eval_r2_final),
        "benefit_eval_r2_eval": float(benefit_eval_r2_eval),
        "benefit_loss_final": float(benefit_loss_final),
        "calibration_gap_benefit": float(calibration_gap_benefit),
        "harm_calibration_gap": float(harm_calibration_gap),
        "selection_bias_benefit": float(mean_selection_bias),
        "benefit_score_rank_pct": float(mean_rank_pct),
        "benefit_vs_harm_ratio": float(benefit_vs_harm_ratio),
        "train_resource_events": int(counts["resource"]),
    }


def run(
    seeds: Tuple = (42, 7),
    warmup_episodes: int = 400,
    eval_episodes: int = 60,
    steps_per_episode: int = 200,
    curriculum_episodes: int = 80,
    self_dim: int = 32,
    world_dim: int = 32,
    lr: float = 1e-3,
    alpha_world: float = 0.9,
    alpha_self: float = 0.3,
    harm_scale: float = 0.02,
    proximity_harm_scale: float = 0.05,
    novelty_bonus_weight: float = 0.1,
    benefit_weight: float = 2.0,
    temperature: float = 1.0,
    **kwargs,
) -> dict:
    """
    BENEFIT_ENABLED (E3 uses trained benefit_eval_head) vs
    BENEFIT_DISABLED (E3 uses harm+residue only).

    ARC-030 test: does trained benefit_eval_head guide E3 toward resources?
    """
    results_enabled:  List[Dict] = []
    results_disabled: List[Dict] = []

    for seed in seeds:
        for benefit_enabled in [True, False]:
            label = "BENEFIT_ENABLED" if benefit_enabled else "BENEFIT_DISABLED"
            print(
                f"\n[V3-EXQ-085m] {label} seed={seed}"
                f" warmup={warmup_episodes} eval={eval_episodes}"
                f" curriculum={curriculum_episodes}"
                f" benefit_weight={benefit_weight} alpha_world={alpha_world}",
                flush=True,
            )
            r = _run_single(
                seed=seed,
                benefit_enabled=benefit_enabled,
                warmup_episodes=warmup_episodes,
                eval_episodes=eval_episodes,
                steps_per_episode=steps_per_episode,
                curriculum_episodes=curriculum_episodes,
                self_dim=self_dim,
                world_dim=world_dim,
                lr=lr,
                alpha_world=alpha_world,
                alpha_self=alpha_self,
                harm_scale=harm_scale,
                proximity_harm_scale=proximity_harm_scale,
                novelty_bonus_weight=novelty_bonus_weight,
                benefit_weight=benefit_weight,
                temperature=temperature,
            )
            if benefit_enabled:
                results_enabled.append(r)
            else:
                results_disabled.append(r)

    def _avg(results: List[Dict], key: str) -> float:
        vals = [r[key] for r in results]
        return float(sum(vals) / max(1, len(vals)))

    benefit_present = _avg(results_enabled,  "avg_benefit_per_ep")
    benefit_absent  = _avg(results_disabled, "avg_benefit_per_ep")
    benefit_ratio = (
        benefit_present / max(1e-6, benefit_absent)
        if benefit_absent > 1e-6 else 0.0
    )
    avg_r2_train       = _avg(results_enabled, "benefit_eval_r2_train")
    avg_r2_eval        = _avg(results_enabled, "benefit_eval_r2_eval")
    avg_cal_gap        = _avg(results_enabled, "calibration_gap_benefit")
    avg_harm_cal_gap   = _avg(results_enabled, "harm_calibration_gap")
    avg_sel_bias       = _avg(results_enabled, "selection_bias_benefit")
    avg_rank_pct       = _avg(results_enabled, "benefit_score_rank_pct")
    avg_bvh            = _avg(results_enabled, "benefit_vs_harm_ratio")

    # PASS criteria
    c1_pass = avg_r2_train > 0.3       # benefit_eval_head learned from z_world
    c2_pass = benefit_ratio >= 1.3     # benefit-guided nav beats harm-only
    c3_pass = avg_sel_bias > 0.005     # E3 selects higher-benefit trajectories
    c4_pass = avg_cal_gap > 0.0        # benefit_eval_head higher near resources

    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass])
    all_pass = criteria_met == 4
    status = "PASS" if all_pass else "FAIL"

    print(f"\n[V3-EXQ-085m] Final results:", flush=True)
    print(
        f"  benefit_enabled={benefit_present:.3f}"
        f"  benefit_disabled={benefit_absent:.3f}"
        f"  ratio={benefit_ratio:.2f}x",
        flush=True,
    )
    print(
        f"  benefit_r2_train={avg_r2_train:.3f}"
        f"  benefit_r2_eval={avg_r2_eval:.3f}"
        f"  cal_gap={avg_cal_gap:.4f}"
        f"  harm_cal_gap={avg_harm_cal_gap:.4f}",
        flush=True,
    )
    print(
        f"  sel_bias={avg_sel_bias:.5f}"
        f"  rank_pct={avg_rank_pct:.3f}"
        f"  benefit_vs_harm={avg_bvh:.3f}",
        flush=True,
    )
    print(f"  status={status} ({criteria_met}/4)", flush=True)

    failure_notes: List[str] = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: benefit_r2_train={avg_r2_train:.3f} <= 0.3."
            " benefit_eval_head cannot learn resource proximity from z_world."
            " Check if z_world encodes resource_field_view (world_dim, alpha_world)."
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: benefit_ratio={benefit_ratio:.2f}x < 1.3x."
            " Benefit-guided E3 not improving navigation vs harm-only E3."
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: selection_bias={avg_sel_bias:.5f} <= 0.005."
            " E3 not preferentially selecting higher-benefit trajectories."
            " H-B CONFIRMED independently of H-A."
        )
    if not c4_pass:
        failure_notes.append(
            f"C4 FAIL: calibration_gap={avg_cal_gap:.4f} <= 0.0."
            " benefit_eval_head not assigning higher scores near resources."
        )

    for note in failure_notes:
        print(f"  {note}", flush=True)

    per_enabled_rows = "\n".join(
        f"  seed={r['seed']}:"
        f" benefit/ep={r['avg_benefit_per_ep']:.3f}"
        f" r2_train={r['benefit_eval_r2_train']:.3f}"
        f" r2_eval={r['benefit_eval_r2_eval']:.3f}"
        f" sel_bias={r['selection_bias_benefit']:.5f}"
        f" rank_pct={r['benefit_score_rank_pct']:.3f}"
        for r in results_enabled
    )
    per_disabled_rows = "\n".join(
        f"  seed={r['seed']}: benefit/ep={r['avg_benefit_per_ep']:.3f}"
        for r in results_disabled
    )

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(
            f"- {n}" for n in failure_notes
        )

    # C3 failure conclusion
    if avg_sel_bias < 0.005:
        hb_conclusion = (
            f"H-B CONFIRMED independently: selection_bias={avg_sel_bias:.5f} < 0.005."
            " E3 does not preferentially select higher-benefit trajectories."
            " Benefit scoring in score_trajectory() not effective."
        )
    elif avg_sel_bias > 0.02:
        hb_conclusion = (
            f"H-B DISCONFIRMED: selection_bias={avg_sel_bias:.5f} > 0.02."
            " E3 IS selecting higher-benefit trajectories."
        )
    else:
        hb_conclusion = (
            f"H-B INCONCLUSIVE: selection_bias={avg_sel_bias:.5f} (0.005-0.02)."
        )

    summary_markdown = (
        f"# V3-EXQ-085m -- ARC-030 benefit_eval_head + E3 Selection\n\n"
        f"**Status:** {status}\n"
        f"**Claims:** ARC-030, MECH-112, SD-015\n"
        f"**Seeds:** {list(seeds)}\n\n"
        f"## Experimental Design\n\n"
        f"First experiment to train `benefit_eval_head` on resource proximity regression\n"
        f"and test whether E3's native benefit pathway drives goal-directed navigation.\n\n"
        f"Training target: `max(resource_field_view)` per step (MSE regression).\n"
        f"Evaluation: BENEFIT_ENABLED (E3 uses benefit) vs BENEFIT_DISABLED (E3 harm+residue only).\n\n"
        f"No goal_state / z_goal attractor. Pure ARC-030 Go-channel test.\n\n"
        f"**benefit_weight:** {benefit_weight}  **alpha_world:** {alpha_world}\n"
        f"**Warmup:** {warmup_episodes} eps (curriculum={curriculum_episodes} eps)\n"
        f"**Eval:** {eval_episodes} eps per condition\n\n"
        f"## Key Results\n\n"
        f"| Metric | Value | Criterion | Status |\n"
        f"|---|---|---|---|\n"
        f"| benefit_eval_r2_train | {avg_r2_train:.3f} | > 0.3 | {'PASS' if c1_pass else 'FAIL'} |\n"
        f"| benefit_ratio | {benefit_ratio:.2f}x | >= 1.3x | {'PASS' if c2_pass else 'FAIL'} |\n"
        f"| selection_bias_benefit | {avg_sel_bias:.5f} | > 0.005 | {'PASS' if c3_pass else 'FAIL'} |\n"
        f"| calibration_gap_benefit | {avg_cal_gap:.4f} | > 0.0 | {'PASS' if c4_pass else 'FAIL'} |\n\n"
        f"## Navigation Results\n\n"
        f"| Condition | benefit/ep |\n"
        f"|---|---|\n"
        f"| BENEFIT_ENABLED | {benefit_present:.3f} |\n"
        f"| BENEFIT_DISABLED | {benefit_absent:.3f} |\n\n"
        f"**Benefit ratio: {benefit_ratio:.2f}x**\n\n"
        f"## H-B Analysis\n\n{hb_conclusion}\n\n"
        f"benefit_eval_r2_train={avg_r2_train:.3f}  "
        f"benefit_eval_r2_eval={avg_r2_eval:.3f}  "
        f"calibration_gap={avg_cal_gap:.4f}  "
        f"harm_calibration_gap={avg_harm_cal_gap:.4f}\n\n"
        f"benefit_vs_harm_ratio={avg_bvh:.3f}\n\n"
        f"## Per-Seed\n\n"
        f"BENEFIT_ENABLED:\n{per_enabled_rows}\n\n"
        f"BENEFIT_DISABLED:\n{per_disabled_rows}\n"
        f"{failure_section}\n"
    )

    metrics = {
        "benefit_per_ep_enabled":       float(benefit_present),
        "benefit_per_ep_disabled":      float(benefit_absent),
        "benefit_ratio":                float(benefit_ratio),
        "benefit_eval_r2_train":        float(avg_r2_train),
        "benefit_eval_r2_eval":         float(avg_r2_eval),
        "calibration_gap_benefit":      float(avg_cal_gap),
        "harm_calibration_gap":         float(avg_harm_cal_gap),
        "selection_bias_benefit":       float(avg_sel_bias),
        "benefit_score_rank_pct":       float(avg_rank_pct),
        "benefit_vs_harm_ratio":        float(avg_bvh),
        "crit1_pass":                   1.0 if c1_pass else 0.0,
        "crit2_pass":                   1.0 if c2_pass else 0.0,
        "crit3_pass":                   1.0 if c3_pass else 0.0,
        "crit4_pass":                   1.0 if c4_pass else 0.0,
        "criteria_met":                 float(criteria_met),
        "n_seeds":                      float(len(seeds)),
        "alpha_world":                  float(alpha_world),
        "benefit_weight":               float(benefit_weight),
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
    parser.add_argument("--seeds",           type=int,   nargs="+", default=[42, 7])
    parser.add_argument("--warmup",          type=int,   default=400)
    parser.add_argument("--eval-eps",        type=int,   default=60)
    parser.add_argument("--steps",           type=int,   default=200)
    parser.add_argument("--curriculum",      type=int,   default=80)
    parser.add_argument("--alpha-world",     type=float, default=0.9)
    parser.add_argument("--alpha-self",      type=float, default=0.3)
    parser.add_argument("--harm-scale",      type=float, default=0.02)
    parser.add_argument("--proximity-scale", type=float, default=0.05)
    parser.add_argument("--novelty-weight",  type=float, default=0.1)
    parser.add_argument("--benefit-weight",  type=float, default=2.0)
    parser.add_argument("--temperature",     type=float, default=1.0)
    parser.add_argument("--dry-run",         action="store_true",
                        help="Run 1 seed, 5 warmup, 5 eval for smoke test")
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
        benefit_weight=args.benefit_weight,
        temperature=args.temperature,
    )

    # Write output
    out_dir = Path(__file__).resolve().parents[2] / "REE_assembly" / "evidence" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    exp_dir = out_dir / EXPERIMENT_TYPE
    exp_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest = {
        "run_id": f"{EXPERIMENT_TYPE}_{run_id}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "status": result["status"],
        "evidence_direction": result["evidence_direction"],
        "metrics": result["metrics"],
        "output_files": [],
    }
    manifest_path = exp_dir / f"{EXPERIMENT_TYPE}_{run_id}.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    summary_path = exp_dir / f"{EXPERIMENT_TYPE}_{run_id}_summary.md"
    with open(summary_path, "w") as f:
        f.write(result["summary_markdown"])

    print(f"\nResults written to {manifest_path}", flush=True)
    print(f"Summary written to {summary_path}", flush=True)
    print(f"Status: {result['status']}", flush=True)
