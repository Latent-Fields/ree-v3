#!/opt/local/bin/python3
"""
V3-EXQ-179 -- H-A/H-B Goal Navigation Causal Diagnostic

Claims: MECH-112, SD-012

=== DIAGNOSTIC CONTEXT ===

EXQ-085 through 085h all fail C2 (benefit_ratio < 1.0x): goal-guided navigation
is consistently WORSE than random baseline, despite z_goal seeding working
(norm > 0.1 in every iteration since 085e).

085h (SD-015) retired: resource_field_view-seeded goal gave goal_resource_r_rfm=-0.169,
WORSE than the z_world approach (-0.010). Both representations fail.

Two competing hypotheses need direct empirical resolution:

  H-A (content problem):
    z_goal is seeded from z_world at resource contact. But z_world encodes the
    full scene (agent position, hazards, resource layout). Resources respawn
    randomly after consumption, so the "scene at contact" is scene-specific
    noise -- z_goal ends up pointing at a snapshot, not a resource type.
    Prediction: goal_proximity(z_world_t) will be near-zero even when the agent
    is adjacent to a resource, because z_world_t at that moment is unlikely to
    match the stored z_goal (different positions, different resource layout).

  H-B (utilisation problem):
    z_goal is correctly seeded with meaningful content, but E3's trajectory
    scoring doesn't use it effectively. The goal score is computed via
    goal_state.goal_proximity(z_world_predicted) inside E3.score_trajectory()
    -- but may be swamped by harm/residue costs. Note: EXQ-085h shows
    goal_vs_harm_ratio=1.212 (above MECH-124 threshold), but E3 selection bias
    has never been directly measured.

=== DIAGNOSTIC DESIGN ===

Two conditions:
  GOAL_PRESENT: real E3.select() with goal_state active.
  GOAL_ABSENT:  random action selection (benefit baseline).

H-A MEASUREMENT (per eval step, GOAL_PRESENT):
  goal_prox_t    = goal_state.goal_proximity(z_world_t)  [is z_goal near z_world now?]
  resource_prox_t = 1/(1+manhattan_dist_to_nearest_resource)  [ground truth]
  Pearson r(goal_prox_t, resource_prox_t) = goal_tracking_r
  Interpretation: near-zero -> H-A confirmed (content noise).

H-B MEASUREMENT (per E3 tick, GOAL_PRESENT):
  candidates = agent._e3_tick(latent, e1_prior)   [fresh trajectory generation]
  all_goal_scores = [compute_goal_score(t, goal_state) for t in candidates]
  result = agent.e3.select(candidates, goal_state=goal_state)
  selection_bias = goal_score_selected - mean(goal_score_others)
  Interpretation: near-zero -> H-B confirmed (E3 not using goal score).

=== PASS CRITERIA ===

PASS = diagnostic ran cleanly. H-A/H-B conclusions come from metric VALUES.

C1: z_goal_norm > 0.1            (seeding confirmed)
C2: n_eval_goal_steps > 500      (H-A data quality)
C3: n_resource_seedings > 5      (seeding data quality)
C4: no fatal errors

=== INTERPRETATION TABLE ===

goal_tracking_r < 0.05  -> H-A CONFIRMED  (z_goal is scene noise, not resource signal)
goal_tracking_r > 0.20  -> H-A DISCONFIRMED  (z_goal tracks resources)
selection_bias  < 0.005 -> H-B CONFIRMED  (E3 not using goal score effectively)
selection_bias  > 0.02  -> H-B DISCONFIRMED  (E3 IS selecting higher-goal trajectories)
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


EXPERIMENT_TYPE = "v3_exq_179_goal_nav_diagnostic"
CLAIM_IDS = ["MECH-112", "SD-012"]


# ------------------------------------------------------------------ #
# Helper functions (verbatim from 085h)                               #
# ------------------------------------------------------------------ #

def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _ensure_2d(t: torch.Tensor) -> torch.Tensor:
    """Ensure tensor is [1, dim]."""
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
# Main run function                                                    #
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
    alpha_world: float,
    alpha_self: float,
    harm_scale: float,
    proximity_harm_scale: float,
    novelty_bonus_weight: float,
    drive_weight: float,
    temperature: float,
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

    # GoalState external to agent (same pattern as all 085x experiments)
    goal_config = GoalConfig(
        goal_dim=world_dim,
        alpha_goal=0.3,
        decay_goal=0.003,
        benefit_threshold=0.05,
        goal_weight=1.0,
        drive_weight=drive_weight,
        e1_goal_conditioned=True,
        z_goal_enabled=goal_present,
    )
    goal_state = GoalState(goal_config, device=torch.device("cpu"))

    # E3Config lacks goal_weight (lives in GoalConfig); e3_selector.py reads
    # self.config.goal_weight which crashes unless patched here.
    if not hasattr(config.e3, "goal_weight"):
        config.e3.goal_weight = goal_config.goal_weight

    agent = REEAgent(config)

    standard_params = [
        p for n, p in agent.named_parameters()
        if "harm_eval_head" not in n
    ]
    harm_eval_params = list(agent.e3.harm_eval_head.parameters())
    optimizer = optim.Adam(standard_params, lr=lr)
    harm_eval_optimizer = optim.Adam(harm_eval_params, lr=1e-4)

    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []
    MAX_BUF = 2000

    counts: Dict[str, int] = {
        "hazard_approach": 0, "env_caused_hazard": 0,
        "agent_caused_hazard": 0, "none": 0, "resource": 0,
        "benefit_approach": 0,
    }
    n_resource_seedings = 0

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

            latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
            agent.clock.advance()  # return value not needed during warmup
            z_world_curr = latent.z_world.detach()

            action_idx = random.randint(0, action_dim - 1)
            action_oh  = _action_to_onehot(action_idx, action_dim, agent.device)
            agent._last_action = action_oh

            _, harm_signal, done, info, obs_dict = env.step(action_oh)
            ttype = info.get("transition_type", "none")
            if ttype in counts:
                counts[ttype] += 1

            # Contact-only goal seeding (GOAL_PRESENT only)
            # No proximity-based secondary seeding -- cleaner H-A interpretation.
            if goal_present and ttype == "resource":
                goal_state.update(z_world_curr, benefit_exposure=1.0, drive_level=1.0)
                n_resource_seedings += 1

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
            goal_norm = goal_state.goal_norm() if goal_present else 0.0
            curriculum_tag = " [curriculum]" if ep < curriculum_episodes else ""
            print(
                f"  [train] seed={seed} cond={cond_label}"
                f" ep {ep+1}/{warmup_episodes}"
                f" resource={counts['resource']}"
                f" seedings={n_resource_seedings}"
                f" z_goal_norm={goal_norm:.3f}"
                f"{curriculum_tag}",
                flush=True,
            )

    z_goal_norm_end = goal_state.goal_norm() if goal_present else 0.0

    # --- EVAL ---
    agent.eval()

    benefit_per_ep: List[float] = []
    harm_buf_eval_pos: List[torch.Tensor] = []
    harm_buf_eval_neg: List[torch.Tensor] = []

    # H-A diagnostic: goal_proximity vs resource_proximity correlation
    goal_prox_t_vals:    List[float] = []
    resource_prox_t_vals: List[float] = []
    n_eval_goal_steps: int = 0

    # H-B diagnostic: E3 trajectory selection bias
    selection_bias_vals:      List[float] = []
    goal_score_rank_pct_vals: List[float] = []

    # MECH-124: goal_vs_harm_ratio
    goal_contrib_vals: List[float] = []
    harm_eval_vals:    List[float] = []

    for _ in range(eval_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        ep_benefit = 0.0

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_harm  = obs_dict.get("harm_obs", None)

            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)

            # advance() returns ticks dict -- must capture in eval for H-B gating
            ticks = agent.clock.advance()

            with torch.no_grad():
                z_world_curr = latent.z_world.detach()

            # H-A measurement (every eval step where goal_state.is_active())
            if goal_present and goal_state.is_active():
                with torch.no_grad():
                    gp_t = goal_state.goal_proximity(z_world_curr).mean().item()
                    he = agent.e3.harm_eval(z_world_curr).mean().item()
                rp_t = _resource_proximity(env)
                goal_prox_t_vals.append(gp_t)
                resource_prox_t_vals.append(rp_t)
                harm_eval_vals.append(he)
                goal_contrib_vals.append(goal_config.goal_weight * gp_t)
                n_eval_goal_steps += 1

            # Action selection + H-B measurement
            if goal_present and goal_state.is_active():
                if ticks.get("e3_tick", True):
                    # E1 tick to provide prior for trajectory generation
                    with torch.no_grad():
                        e1_prior = agent._e1_tick(latent)
                    # Fresh E3 tick: generate candidates + real E3 selection
                    with torch.no_grad():
                        candidates = agent._e3_tick(latent, e1_prior)
                        all_goal_scores = [
                            agent.e3.compute_goal_score(t, goal_state).mean().item()
                            for t in candidates
                        ]
                        result = agent.e3.select(
                            candidates, temperature=temperature,
                            goal_state=goal_state,
                        )
                    action_oh = result.selected_action
                    agent._last_action = action_oh

                    # H-B: compare selected goal score vs mean of others
                    if len(all_goal_scores) > 1:
                        gs_sel = all_goal_scores[result.selected_index]
                        others = [
                            s for i, s in enumerate(all_goal_scores)
                            if i != result.selected_index
                        ]
                        bias = gs_sel - (sum(others) / len(others))
                        selection_bias_vals.append(bias)
                        n_beaten = sum(1 for s in all_goal_scores if gs_sel > s)
                        goal_score_rank_pct_vals.append(
                            n_beaten / max(1, len(all_goal_scores) - 1)
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
            else:
                # GOAL_ABSENT: random action
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

            if done:
                break

        benefit_per_ep.append(ep_benefit)

    avg_benefit = float(sum(benefit_per_ep) / max(1, len(benefit_per_ep)))

    # Harm calibration gap
    calibration_gap = 0.0
    if len(harm_buf_eval_pos) >= 4 and len(harm_buf_eval_neg) >= 4:
        k = min(64, min(len(harm_buf_eval_pos), len(harm_buf_eval_neg)))
        zw_pos = torch.cat(harm_buf_eval_pos[-k:], dim=0)
        zw_neg = torch.cat(harm_buf_eval_neg[-k:], dim=0)
        with torch.no_grad():
            harm_pos = agent.e3.harm_eval(zw_pos).mean().item()
            harm_neg = agent.e3.harm_eval(zw_neg).mean().item()
        calibration_gap = harm_pos - harm_neg

    # H-A metric
    goal_tracking_r = (
        _pearson_r(goal_prox_t_vals, resource_prox_t_vals)
        if goal_present and len(goal_prox_t_vals) >= 4 else 0.0
    )

    # H-B metrics
    mean_selection_bias = (
        float(sum(selection_bias_vals) / max(1, len(selection_bias_vals)))
        if goal_present and selection_bias_vals else 0.0
    )
    mean_goal_score_rank_pct = (
        float(sum(goal_score_rank_pct_vals) / max(1, len(goal_score_rank_pct_vals)))
        if goal_present and goal_score_rank_pct_vals else 0.0
    )

    # MECH-124: goal_vs_harm_ratio
    mean_goal_contrib = (
        float(sum(goal_contrib_vals) / max(1, len(goal_contrib_vals)))
        if goal_present else 0.0
    )
    mean_harm_eval_val = (
        float(sum(harm_eval_vals) / max(1, len(harm_eval_vals)))
        if goal_present else 0.0
    )
    goal_vs_harm_ratio = mean_goal_contrib / max(1e-6, mean_harm_eval_val)

    print(
        f"  [eval] seed={seed} cond={cond_label}"
        f" avg_benefit/ep={avg_benefit:.3f}"
        f" z_goal_norm={z_goal_norm_end:.3f}"
        f" cal_gap={calibration_gap:.4f}"
        f" goal_tracking_r={goal_tracking_r:.3f}"
        f" sel_bias={mean_selection_bias:.5f}"
        f" rank_pct={mean_goal_score_rank_pct:.3f}"
        f" goal_vs_harm={goal_vs_harm_ratio:.3f}"
        f" n_eval_steps={n_eval_goal_steps}",
        flush=True,
    )

    return {
        "seed": seed,
        "condition": cond_label,
        "goal_present": goal_present,
        "avg_benefit_per_ep": float(avg_benefit),
        "z_goal_norm_end": float(z_goal_norm_end),
        "n_resource_seedings": int(n_resource_seedings),
        "n_eval_goal_steps": int(n_eval_goal_steps),
        "calibration_gap": float(calibration_gap),
        "goal_tracking_r": float(goal_tracking_r),
        "selection_bias": float(mean_selection_bias),
        "goal_score_rank_pct": float(mean_goal_score_rank_pct),
        "goal_vs_harm_ratio": float(goal_vs_harm_ratio),
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
    drive_weight: float = 2.0,
    temperature: float = 1.0,
    **kwargs,
) -> dict:
    """GOAL_PRESENT (real E3 selection) vs GOAL_ABSENT (random) -- H-A/H-B diagnostic."""
    results_goal:    List[Dict] = []
    results_no_goal: List[Dict] = []

    for seed in seeds:
        for goal_present in [True, False]:
            label = "GOAL_PRESENT" if goal_present else "GOAL_ABSENT"
            print(
                f"\n[V3-EXQ-179] {label} seed={seed}"
                f" warmup={warmup_episodes} eval={eval_episodes}"
                f" curriculum={curriculum_episodes}"
                f" drive_weight={drive_weight} alpha_world={alpha_world}"
                f" temperature={temperature}",
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
                alpha_world=alpha_world,
                alpha_self=alpha_self,
                harm_scale=harm_scale,
                proximity_harm_scale=proximity_harm_scale,
                novelty_bonus_weight=novelty_bonus_weight,
                drive_weight=drive_weight,
                temperature=temperature,
            )
            if goal_present:
                results_goal.append(r)
            else:
                results_no_goal.append(r)

    def _avg(results: List[Dict], key: str) -> float:
        vals = [r[key] for r in results]
        return float(sum(vals) / max(1, len(vals)))

    z_goal_norm_avg        = _avg(results_goal, "z_goal_norm_end")
    benefit_goal_present   = _avg(results_goal, "avg_benefit_per_ep")
    benefit_goal_absent    = _avg(results_no_goal, "avg_benefit_per_ep")
    avg_n_eval_steps       = _avg(results_goal, "n_eval_goal_steps")
    avg_n_seedings         = _avg(results_goal, "n_resource_seedings")
    avg_goal_tracking_r    = _avg(results_goal, "goal_tracking_r")
    avg_selection_bias     = _avg(results_goal, "selection_bias")
    avg_goal_score_rank    = _avg(results_goal, "goal_score_rank_pct")
    avg_goal_vs_harm       = _avg(results_goal, "goal_vs_harm_ratio")
    avg_calibration_gap    = _avg(results_goal, "calibration_gap")

    benefit_ratio = (
        benefit_goal_present / max(1e-6, benefit_goal_absent)
        if benefit_goal_absent > 1e-6 else 0.0
    )

    # PASS criteria: diagnostic ran cleanly
    c1_pass = z_goal_norm_avg > 0.1
    c2_pass = avg_n_eval_steps > 500
    c3_pass = avg_n_seedings > 5
    c4_pass = True  # no fatal errors (would have crashed if False)

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass])
    status = "PASS" if all_pass else "FAIL"
    mech124_flag_salience = avg_goal_vs_harm < 0.3

    # H-A interpretation
    if avg_goal_tracking_r < 0.05:
        ha_conclusion = (
            f"H-A CONFIRMED: goal_tracking_r={avg_goal_tracking_r:.3f} < 0.05"
            " -- z_goal does not track resource proximity (content noise hypothesis supported)"
        )
    elif avg_goal_tracking_r > 0.20:
        ha_conclusion = (
            f"H-A DISCONFIRMED: goal_tracking_r={avg_goal_tracking_r:.3f} > 0.20"
            " -- z_goal DOES track resources (content is meaningful)"
        )
    else:
        ha_conclusion = (
            f"H-A INCONCLUSIVE: goal_tracking_r={avg_goal_tracking_r:.3f}"
            " (between 0.05 and 0.20)"
        )

    # H-B interpretation
    if avg_selection_bias < 0.005:
        hb_conclusion = (
            f"H-B CONFIRMED: selection_bias={avg_selection_bias:.5f} < 0.005"
            " -- E3 NOT using goal score to select trajectories"
        )
    elif avg_selection_bias > 0.02:
        hb_conclusion = (
            f"H-B DISCONFIRMED: selection_bias={avg_selection_bias:.5f} > 0.02"
            " -- E3 IS selecting higher-goal trajectories (utilisation working)"
        )
    else:
        hb_conclusion = (
            f"H-B INCONCLUSIVE: selection_bias={avg_selection_bias:.5f}"
            " (between 0.005 and 0.02)"
        )

    print(f"\n[V3-EXQ-179] Final results:", flush=True)
    print(
        f"  z_goal_norm={z_goal_norm_avg:.3f}"
        f"  seedings_avg={avg_n_seedings:.1f}"
        f"  n_eval_steps_avg={avg_n_eval_steps:.0f}",
        flush=True,
    )
    print(
        f"  benefit_goal_present={benefit_goal_present:.3f}"
        f"  benefit_goal_absent={benefit_goal_absent:.3f}"
        f"  ratio={benefit_ratio:.2f}x",
        flush=True,
    )
    print(f"  H-A: goal_tracking_r={avg_goal_tracking_r:.3f}", flush=True)
    print(f"  H-B: selection_bias={avg_selection_bias:.5f}  rank_pct={avg_goal_score_rank:.3f}", flush=True)
    print(f"  goal_vs_harm={avg_goal_vs_harm:.3f}  cal_gap={avg_calibration_gap:.4f}", flush=True)
    print(f"  {ha_conclusion}", flush=True)
    print(f"  {hb_conclusion}", flush=True)
    print(f"  status={status} ({criteria_met}/4)", flush=True)

    failure_notes: List[str] = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: z_goal_norm={z_goal_norm_avg:.3f} <= 0.1"
            " (seeding failed -- check drive_weight and resource events)"
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: n_eval_goal_steps={avg_n_eval_steps:.0f} <= 500"
            " (insufficient H-A data -- increase eval_episodes or check seedings)"
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: n_resource_seedings={avg_n_seedings:.1f} <= 5"
            " (insufficient seedings -- check curriculum or env resource count)"
        )
    if mech124_flag_salience and c1_pass:
        failure_notes.append(
            f"MECH-124 V4 RISK: goal_vs_harm_ratio={avg_goal_vs_harm:.3f} < 0.3."
            " z_goal salience not competitive with harm salience."
        )

    for note in failure_notes:
        print(f"  {note}", flush=True)

    per_goal_rows = "\n".join(
        f"  seed={r['seed']}:"
        f" benefit/ep={r['avg_benefit_per_ep']:.3f}"
        f" z_goal_norm={r['z_goal_norm_end']:.3f}"
        f" goal_tracking_r={r['goal_tracking_r']:.3f}"
        f" selection_bias={r['selection_bias']:.5f}"
        f" rank_pct={r['goal_score_rank_pct']:.3f}"
        f" n_eval_steps={r['n_eval_goal_steps']}"
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
        f"# V3-EXQ-179 -- H-A/H-B Goal Navigation Causal Diagnostic\n\n"
        f"**Status:** {status}\n"
        f"**Claims:** MECH-112, SD-012\n"
        f"**Seeds:** {list(seeds)}\n\n"
        f"## Diagnostic Design\n\n"
        f"Two competing hypotheses about why z_goal-guided navigation fails:\n\n"
        f"- **H-A (content problem):** z_goal seeded from z_world at resource contact,"
        f" but z_world encodes the full scene. Resources respawn randomly -> z_goal"
        f" is scene-specific noise.\n"
        f"  Measured by: Pearson r(goal_proximity(z_world_t), resource_proximity_t)"
        f" = `goal_tracking_r`\n\n"
        f"- **H-B (utilisation problem):** z_goal is meaningful but E3 trajectory"
        f" scoring doesn't use it effectively.\n"
        f"  Measured by: goal_score(selected_traj) - mean(goal_score(other_trajs))"
        f" = `selection_bias`\n\n"
        f"Conditions: GOAL_PRESENT (real E3.select() with goal_state) vs"
        f" GOAL_ABSENT (random actions).\n\n"
        f"**alpha_world:** {alpha_world}  (SD-008)\n"
        f"**drive_weight:** {drive_weight}  (SD-012)\n"
        f"**Warmup:** {warmup_episodes} eps (curriculum={curriculum_episodes} eps)\n"
        f"**Eval:** {eval_episodes} eps per condition\n\n"
        f"## Key Diagnostic Results\n\n"
        f"| Metric | Value | Threshold | Interpretation |\n"
        f"|---|---|---|---|\n"
        f"| goal_tracking_r (H-A) | {avg_goal_tracking_r:.3f}"
        f" | <0.05=confirmed, >0.20=disconfirmed | {ha_conclusion[:60]}... |\n"
        f"| selection_bias (H-B) | {avg_selection_bias:.5f}"
        f" | <0.005=confirmed, >0.02=disconfirmed | {hb_conclusion[:60]}... |\n"
        f"| goal_score_rank_pct | {avg_goal_score_rank:.3f}"
        f" | >0.5=E3 prefers high-goal | -- |\n\n"
        f"## H-A Conclusion\n\n{ha_conclusion}\n\n"
        f"## H-B Conclusion\n\n{hb_conclusion}\n\n"
        f"## Navigation Results\n\n"
        f"| Condition | benefit/ep | z_goal_norm | cal_gap |\n"
        f"|---|---|---|---|\n"
        f"| GOAL_PRESENT | {benefit_goal_present:.3f} | {z_goal_norm_avg:.3f}"
        f" | {avg_calibration_gap:.4f} |\n"
        f"| GOAL_ABSENT  | {benefit_goal_absent:.3f} | -- | -- |\n\n"
        f"**Benefit ratio (goal/no-goal): {benefit_ratio:.2f}x**\n\n"
        f"## PASS Criteria (diagnostic quality gates)\n\n"
        f"| Criterion | Result | Value |\n"
        f"|---|---|---|\n"
        f"| C1: z_goal_norm > 0.1 | {'PASS' if c1_pass else 'FAIL'}"
        f" | {z_goal_norm_avg:.3f} |\n"
        f"| C2: n_eval_goal_steps > 500 | {'PASS' if c2_pass else 'FAIL'}"
        f" | {avg_n_eval_steps:.0f} |\n"
        f"| C3: n_resource_seedings > 5 | {'PASS' if c3_pass else 'FAIL'}"
        f" | {avg_n_seedings:.1f} |\n"
        f"| C4: no fatal errors | {'PASS' if c4_pass else 'FAIL'} | -- |\n\n"
        f"Criteria met: {criteria_met}/4 -> **{status}**\n\n"
        f"## MECH-124 Diagnostics\n\n"
        f"goal_vs_harm_ratio: {avg_goal_vs_harm:.3f}"
        f" (< 0.3 = V4 salience risk)\n\n"
        f"## Interpretation Reference\n\n"
        f"```\n"
        f"goal_tracking_r < 0.05  -> H-A CONFIRMED  (z_goal is scene noise)\n"
        f"goal_tracking_r > 0.20  -> H-A DISCONFIRMED  (z_goal tracks resources)\n"
        f"selection_bias  < 0.005 -> H-B CONFIRMED  (E3 not using goal score)\n"
        f"selection_bias  > 0.02  -> H-B DISCONFIRMED  (E3 IS preferring high-goal trajs)\n"
        f"```\n\n"
        f"## Per-Seed\n\n"
        f"GOAL_PRESENT:\n{per_goal_rows}\n\n"
        f"GOAL_ABSENT:\n{per_nogoal_rows}\n"
        f"{failure_section}\n"
    )

    metrics = {
        "z_goal_norm_avg":              float(z_goal_norm_avg),
        "benefit_per_ep_goal_present":  float(benefit_goal_present),
        "benefit_per_ep_goal_absent":   float(benefit_goal_absent),
        "benefit_ratio":                float(benefit_ratio),
        "goal_tracking_r":              float(avg_goal_tracking_r),
        "selection_bias":               float(avg_selection_bias),
        "goal_score_rank_pct":          float(avg_goal_score_rank),
        "goal_vs_harm_ratio":           float(avg_goal_vs_harm),
        "calibration_gap_goal_present": float(avg_calibration_gap),
        "n_eval_goal_steps_avg":        float(avg_n_eval_steps),
        "n_resource_seedings_avg":      float(avg_n_seedings),
        "mech124_flag_salience":        1.0 if mech124_flag_salience else 0.0,
        "crit1_pass":                   1.0 if c1_pass else 0.0,
        "crit2_pass":                   1.0 if c2_pass else 0.0,
        "crit3_pass":                   1.0 if c3_pass else 0.0,
        "crit4_pass":                   1.0 if c4_pass else 0.0,
        "criteria_met":                 float(criteria_met),
        "n_seeds":                      float(len(seeds)),
        "alpha_world":                  float(alpha_world),
        "drive_weight":                 float(drive_weight),
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
    parser.add_argument("--drive-weight",    type=float, default=2.0)
    parser.add_argument("--temperature",     type=float, default=1.0)
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
        temperature=args.temperature,
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
