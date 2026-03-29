#!/opt/local/bin/python3
"""
V3-EXQ-085g -- MECH-071 / MECH-112 / MECH-117 / SD-012 Contact-Gated Goal Seeding

Claims: MECH-071, MECH-112, MECH-117, SD-012

Root cause of EXQ-085f C2 FAIL (benefit_ratio=0.28x -- goal-guided WORSE than random):

EXQ-085f PASSED C1 (z_goal_norm=0.228): goal is seeded.
EXQ-085f PASSED C3 (cal_gap=0.030): harm calibration intact.
EXQ-085f FAILED C2 (benefit_ratio=0.28x): goal-guided navigation is 3.5x WORSE than random.
  Diagnostic: goal_resource_r=0.087 (< 0.2 threshold).
  z_goal is seeded but pointing at the WRONG z_world states.

ROOT CAUSE (seeding quality):
085f seeded z_goal on the PROXIMITY EMA signal (benefit_exposure from obs_body[11]).
This fires whenever the agent is anywhere near a resource -- a broad, noisy field.
z_goal accumulates a random cloud of z_world snapshots from various non-resource
positions. E2 lookahead then navigates toward this arbitrary attractor, which has
near-zero correlation with actual resource locations (goal_resource_r=0.087).
Result: goal guidance actively avoids resources by following a spurious attractor.

THE FIX (contact-gated seeding):
Seed z_goal ONLY on actual resource contact events (ttype == "resource"), using a
synthetic benefit_exposure=1.0. This anchors z_goal to z_world states observed at
the moment of resource acquisition -- the most specific and consistent representation
of "resource-like world state" available.
  - alpha_goal raised from 0.1 to 0.3 (faster convergence per contact event;
    contact events are rarer than proximity events, so each must contribute more)
  - Proximity seeding kept as secondary fallback to prevent z_goal decay between
    contact events, but with drive modulation (SD-012) only

MECH-124 diagnostics retained from 085f:
  goal_resource_correlation  -- Pearson r between goal_proximity and resource proximity
  goal_vs_harm_cost_ratio    -- goal salience vs harm salience

PASS criteria (ALL required):
  C1: z_goal_norm_avg > 0.1  (goal seeded)
  C2: benefit_per_ep_goal_present > benefit_per_ep_goal_absent * 1.3  (30% improvement)
  C3: calibration_gap_goal_present > 0.02  (E3 harm calibration intact)
  C4: no fatal errors

Supersedes: V3-EXQ-085f
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
from ree_core.goal import GoalState, GoalConfig
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_085g_mech071_contact_gated_seeding"
CLAIM_IDS = ["MECH-071", "MECH-112", "MECH-117", "SD-012"]


def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


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


def _goal_guided_action(
    agent: REEAgent,
    goal_state: GoalState,
    z_world_curr: torch.Tensor,
    num_actions: int,
    device,
) -> int:
    """
    Pick the action that maximises goal_proximity(E2.world_forward(z_world, action)).

    E2 world_forward provides a single-step prediction of z_world under each candidate
    action. goal_state.goal_proximity() scores that prediction against the current z_goal
    attractor. The action with the highest predicted proximity is selected.

    Falls back to random if z_goal is inactive (norm near zero).
    """
    if not goal_state.is_active():
        return random.randint(0, num_actions - 1)
    best_action = 0
    best_prox = -1.0
    with torch.no_grad():
        for idx in range(num_actions):
            a = _action_to_onehot(idx, num_actions, device)
            z_w_pred = agent.e2.world_forward(z_world_curr, a)
            prox = goal_state.goal_proximity(z_w_pred).mean().item()
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
) -> Dict:
    """Run one (seed, condition) cell."""
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

    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=alpha_world,
        alpha_self=alpha_self,
        reafference_action_dim=0,
        novelty_bonus_weight=novelty_bonus_weight,
    )

    goal_config = GoalConfig(
        goal_dim=world_dim,
        alpha_goal=0.3,          # raised from 0.1: faster convergence per contact event
        decay_goal=0.003,
        benefit_threshold=0.05,
        goal_weight=1.0,
        drive_weight=drive_weight,
        e1_goal_conditioned=True,
        z_goal_enabled=goal_present,
    )
    goal_state = GoalState(goal_config, device=torch.device("cpu"))

    agent = REEAgent(config)

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

    counts: Dict[str, int] = {
        "hazard_approach": 0, "env_caused_hazard": 0,
        "agent_caused_hazard": 0, "none": 0, "resource": 0,
        "benefit_approach": 0,
    }
    total_benefit_warmup = 0.0
    z_goal_norms: List[float] = []
    n_resource_seedings: int = 0  # contact-gated seeding events

    # --- WARMUP TRAINING (random actions for both conditions) ---
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
            agent.clock.advance()

            z_world_curr = latent.z_world.detach()

            # Warmup: random actions (stable training for both conditions)
            action_idx = random.randint(0, env.action_dim - 1)
            action = _action_to_onehot(action_idx, env.action_dim, agent.device)
            agent._last_action = action

            _, harm_signal, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")
            if ttype in counts:
                counts[ttype] += 1

            # --- Goal seeding (GOAL_PRESENT only) ---
            if goal_present:
                if ttype == "resource":
                    # PRIMARY: contact-gated seeding (high-quality anchor).
                    # z_world_curr is the state from BEFORE the step -- the observation
                    # the agent held when it chose to move onto the resource.
                    # benefit_exposure=1.0 forces threshold crossing regardless of drive.
                    goal_state.update(z_world_curr, benefit_exposure=1.0, drive_level=1.0)
                    n_resource_seedings += 1
                    total_benefit_warmup += 1.0
                elif obs_body.shape[-1] > 11:
                    # SECONDARY: proximity-based seeding (low quality, prevents decay).
                    # SD-012 drive modulation: hungry agent gets higher effective benefit.
                    benefit_exposure = float(
                        obs_body[0, 11].item() if obs_body.dim() == 2
                        else obs_body[11].item()
                    )
                    energy = float(
                        obs_body[0, 3].item() if obs_body.dim() == 2
                        else obs_body[3].item()
                    )
                    drive_level = max(0.0, 1.0 - energy)
                    goal_state.update(z_world_curr, benefit_exposure,
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

        if goal_present:
            z_goal_norms.append(goal_state.goal_norm())

        if (ep + 1) % 100 == 0 or ep == warmup_episodes - 1:
            goal_norm_str = (
                f" z_goal_norm={goal_state.goal_norm():.3f}"
                f" n_resource_seedings={n_resource_seedings}"
                if goal_present else ""
            )
            curriculum_tag = " [curriculum]" if ep < curriculum_episodes else ""
            print(
                f"  [train] seed={seed} cond={cond_label}"
                f" ep {ep+1}/{warmup_episodes}"
                f" resource={counts['resource']}"
                f" benefit_approach={counts['benefit_approach']}"
                f" contact={counts['env_caused_hazard']+counts['agent_caused_hazard']}"
                f"{goal_norm_str}{curriculum_tag}",
                flush=True,
            )

    z_goal_norm_end = goal_state.goal_norm() if goal_present else 0.0

    # --- EVAL: goal-guided navigation in GOAL_PRESENT, random in GOAL_ABSENT ---
    agent.eval()

    benefit_per_ep: List[float] = []
    harm_buf_eval_pos: List[torch.Tensor] = []
    harm_buf_eval_neg: List[torch.Tensor] = []

    # Diagnostic accumulators (GOAL_PRESENT only)
    goal_prox_vals: List[float] = []
    resource_prox_vals: List[float] = []
    harm_eval_vals: List[float] = []
    goal_contrib_vals: List[float] = []

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
                agent.clock.advance()
                z_world_curr = latent.z_world.detach()

            # Goal-guided action selection in GOAL_PRESENT; random in GOAL_ABSENT
            if goal_present and goal_state.is_active():
                action_idx = _goal_guided_action(
                    agent, goal_state, z_world_curr,
                    env.action_dim, agent.device,
                )
            else:
                action_idx = random.randint(0, env.action_dim - 1)

            action = _action_to_onehot(action_idx, env.action_dim, agent.device)
            agent._last_action = action

            _, harm_signal, done, info, obs_dict = env.step(action)
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
            if goal_present and goal_state.is_active():
                with torch.no_grad():
                    gp = goal_state.goal_proximity(z_world_ev).mean().item()
                    he = agent.e3.harm_eval(z_world_ev).mean().item()
                rp = _resource_proximity(env)
                goal_prox_vals.append(gp)
                resource_prox_vals.append(rp)
                harm_eval_vals.append(he)
                goal_contrib_vals.append(
                    goal_config.goal_weight * gp
                )

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

    # goal_resource_correlation (MECH-124 diagnostic: is z_goal pointing toward resources?)
    goal_resource_r = _pearson_r(goal_prox_vals, resource_prox_vals) if goal_present else 0.0

    # goal_vs_harm_cost_ratio (MECH-124: is z_goal competitive with harm salience?)
    mean_goal_contrib = (
        float(sum(goal_contrib_vals) / max(1, len(goal_contrib_vals)))
        if goal_present else 0.0
    )
    mean_harm_eval = (
        float(sum(harm_eval_vals) / max(1, len(harm_eval_vals)))
        if goal_present else 0.0
    )
    goal_vs_harm_ratio = (
        mean_goal_contrib / max(1e-6, mean_harm_eval)
        if goal_present else 0.0
    )

    print(
        f"  [eval] seed={seed} cond={cond_label}"
        f" avg_benefit/ep={avg_benefit:.3f}"
        f" z_goal_norm={z_goal_norm_end:.3f}"
        f" n_resource_seedings={n_resource_seedings}"
        f" cal_gap={calibration_gap:.4f}"
        f" goal_resource_r={goal_resource_r:.3f}"
        f" goal_vs_harm={goal_vs_harm_ratio:.3f}",
        flush=True,
    )

    return {
        "seed": seed,
        "condition": cond_label,
        "goal_present": goal_present,
        "avg_benefit_per_ep": float(avg_benefit),
        "z_goal_norm_end": float(z_goal_norm_end),
        "n_resource_seedings": int(n_resource_seedings),
        "calibration_gap": float(calibration_gap),
        "goal_resource_r": float(goal_resource_r),
        "goal_vs_harm_ratio": float(goal_vs_harm_ratio),
        "train_resource_events": int(counts["resource"]),
        "train_benefit_approach": int(counts["benefit_approach"]),
        "train_contact_events": int(
            counts["env_caused_hazard"] + counts["agent_caused_hazard"]
        ),
        "z_goal_norm_trajectory": z_goal_norms,
    }


def run(
    seeds: Tuple = (42, 7, 13),
    warmup_episodes: int = 500,
    eval_episodes: int = 100,
    steps_per_episode: int = 200,
    curriculum_episodes: int = 100,
    self_dim: int = 32,
    world_dim: int = 32,
    lr: float = 1e-3,
    alpha_world: float = 0.9,
    alpha_self: float = 0.3,
    harm_scale: float = 0.02,
    proximity_harm_scale: float = 0.05,
    novelty_bonus_weight: float = 0.1,
    drive_weight: float = 2.0,
    **kwargs,
) -> dict:
    """GOAL_PRESENT (goal-guided eval) vs GOAL_ABSENT (random eval)."""
    results_goal:    List[Dict] = []
    results_no_goal: List[Dict] = []

    for seed in seeds:
        for goal_present in [True, False]:
            label = "GOAL_PRESENT" if goal_present else "GOAL_ABSENT"
            print(
                f"\n[V3-EXQ-085g] {label} seed={seed}"
                f" warmup={warmup_episodes} eval={eval_episodes}"
                f" curriculum={curriculum_episodes}"
                f" drive_weight={drive_weight} alpha_world={alpha_world}"
                f" alpha_goal=0.3 (contact-gated)",
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
            )
            if goal_present:
                results_goal.append(r)
            else:
                results_no_goal.append(r)

    def _avg(results: List[Dict], key: str) -> float:
        vals = [r[key] for r in results]
        return float(sum(vals) / max(1, len(vals)))

    z_goal_norm_avg          = _avg(results_goal,    "z_goal_norm_end")
    benefit_goal_present     = _avg(results_goal,    "avg_benefit_per_ep")
    benefit_goal_absent      = _avg(results_no_goal, "avg_benefit_per_ep")
    cal_gap_goal_present     = _avg(results_goal,    "calibration_gap")
    avg_goal_resource_r      = _avg(results_goal,    "goal_resource_r")
    avg_goal_vs_harm_ratio   = _avg(results_goal,    "goal_vs_harm_ratio")
    avg_n_resource_seedings  = _avg(results_goal,    "n_resource_seedings")

    benefit_ratio = (
        benefit_goal_present / max(1e-6, benefit_goal_absent)
        if benefit_goal_absent > 1e-6 else 0.0
    )

    c1_pass = z_goal_norm_avg > 0.1
    c2_pass = benefit_ratio >= 1.3
    c3_pass = cal_gap_goal_present > 0.02
    c4_pass = True

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass])
    status = "PASS" if all_pass else "FAIL"

    if all_pass:
        decision = "retain_ree"
    elif c1_pass and c2_pass:
        decision = "hybridize"
    else:
        decision = "retire_ree_claim"

    # MECH-124 diagnostic flags (not PASS criteria, informational)
    mech124_flag_goal_resource = avg_goal_resource_r < 0.2
    mech124_flag_salience      = avg_goal_vs_harm_ratio < 0.3

    print(f"\n[V3-EXQ-085g] Final results:", flush=True)
    print(
        f"  z_goal_norm_avg={z_goal_norm_avg:.3f}"
        f"  benefit_goal_present={benefit_goal_present:.3f}"
        f"  benefit_goal_absent={benefit_goal_absent:.3f}"
        f"  ratio={benefit_ratio:.2f}x",
        flush=True,
    )
    print(
        f"  cal_gap_goal_present={cal_gap_goal_present:.4f}"
        f"  goal_resource_r={avg_goal_resource_r:.3f}"
        f"  goal_vs_harm_ratio={avg_goal_vs_harm_ratio:.3f}"
        f"  avg_resource_seedings={avg_n_resource_seedings:.0f}",
        flush=True,
    )
    print(
        f"  drive_weight={drive_weight}"
        f"  decision={decision}  status={status} ({criteria_met}/4)",
        flush=True,
    )

    failure_notes: List[str] = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: z_goal_norm={z_goal_norm_avg:.3f} <= 0.1"
            " (goal not seeded -- check resource_seedings > 0 in warmup)"
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: benefit_ratio={benefit_ratio:.2f}x < 1.3x"
            f" (goal_present={benefit_goal_present:.3f}"
            f" vs goal_absent={benefit_goal_absent:.3f})"
        )
        if mech124_flag_goal_resource:
            failure_notes.append(
                f"  -> goal_resource_r={avg_goal_resource_r:.3f} < 0.2:"
                " z_goal still not pointing toward resources despite contact-gated seeding."
                " z_world at resource contact may be too noisy/low-dimensional to encode"
                " location. Next step: dedicated z_resource representation."
            )
        else:
            failure_notes.append(
                f"  -> goal_resource_r={avg_goal_resource_r:.3f} >= 0.2:"
                " z_goal points toward resources but benefit not crossing 1.3x."
                " Increase eval episodes or goal_weight."
            )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: calibration_gap={cal_gap_goal_present:.4f} <= 0.02"
        )
    if mech124_flag_salience and c1_pass:
        failure_notes.append(
            f"MECH-124 V4 RISK: goal_vs_harm_ratio={avg_goal_vs_harm_ratio:.3f} < 0.3."
            " z_goal salience not competitive with harm -- consolidation-mediated"
            " option-space contraction risk in V4. Consider increasing goal_weight."
        )

    for note in failure_notes:
        print(f"  {note}", flush=True)

    per_goal_rows = "\n".join(
        f"  seed={r['seed']}:"
        f" benefit/ep={r['avg_benefit_per_ep']:.3f}"
        f" z_goal_norm={r['z_goal_norm_end']:.3f}"
        f" n_resource_seedings={r['n_resource_seedings']}"
        f" cal_gap={r['calibration_gap']:.4f}"
        f" goal_resource_r={r['goal_resource_r']:.3f}"
        f" goal_vs_harm={r['goal_vs_harm_ratio']:.3f}"
        for r in results_goal
    )
    per_nogoal_rows = "\n".join(
        f"  seed={r['seed']}:"
        f" benefit/ep={r['avg_benefit_per_ep']:.3f}"
        f" resource_events={r['train_resource_events']}"
        for r in results_no_goal
    )

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(
            f"- {n}" for n in failure_notes
        )

    summary_markdown = (
        f"# V3-EXQ-085g -- MECH-071/112/117 Contact-Gated Goal Seeding\n\n"
        f"**Status:** {status}\n"
        f"**Claims:** MECH-071, MECH-112, MECH-117, SD-012\n"
        f"**Decision:** {decision}\n"
        f"**Seeds:** {list(seeds)}\n"
        f"**alpha_world:** {alpha_world}  (SD-008)\n"
        f"**alpha_goal:** 0.3  (raised from 0.1 -- faster convergence per contact event)\n"
        f"**SD-010:** use_proxy_fields=True (harm_obs wired)\n"
        f"**SD-012:** drive_weight={drive_weight}, resource_respawn_on_consume=True\n"
        f"**Curriculum:** first {curriculum_episodes} episodes place resource near spawn\n"
        f"**Warmup:** {warmup_episodes} eps (random actions, same for both conditions)\n"
        f"**Eval:** {eval_episodes} eps (GOAL_PRESENT: E2-lookahead goal guidance;"
        f" GOAL_ABSENT: random)\n"
        f"**Supersedes:** V3-EXQ-085f\n\n"
        f"## Root Cause Fix (Seeding Quality)\n\n"
        f"EXQ-085f had z_goal_norm=0.228 (seeded) but benefit_ratio=0.28x (goal guidance"
        f" WORSE than random). Diagnostic: goal_resource_r=0.087 -- z_goal seeded to random"
        f" z_world states from noisy proximity signal, not resource-specific representations.\n\n"
        f"Fix: PRIMARY seeding now triggered only on actual resource contact events"
        f" (ttype=='resource'), using benefit_exposure=1.0 to force threshold crossing."
        f" This anchors z_goal to z_world observed at the moment of resource acquisition."
        f" Proximity seeding retained as secondary (prevents decay between contacts).\n\n"
        f"avg_resource_seedings_warmup: {avg_n_resource_seedings:.0f}\n\n"
        f"## MECH-124 Diagnostics\n\n"
        f"**goal_resource_correlation (Pearson r):** {avg_goal_resource_r:.3f}\n"
        f"  > 0.2: z_goal points toward resources (contact-gated seeding working)\n"
        f"  < 0.2: z_world at contact too noisy -- need dedicated z_resource\n\n"
        f"**goal_vs_harm_cost_ratio:** {avg_goal_vs_harm_ratio:.3f}\n"
        f"  MECH-124 V4 risk: < 0.3 = z_goal salience not competitive with harm.\n\n"
        f"## Pre-Registered Thresholds\n\n"
        f"C1: z_goal_norm > 0.1 (goal seeded)\n"
        f"C2: benefit_goal_present > benefit_goal_absent * 1.3\n"
        f"C3: calibration_gap_goal_present > 0.02\n"
        f"C4: no fatal errors\n\n"
        f"## Results\n\n"
        f"| Condition | benefit/ep | z_goal_norm | cal_gap | goal_resource_r |"
        f" goal_vs_harm |\n"
        f"|-----------|-----------|-------------|--------|----------------|"
        f"-------------|\n"
        f"| GOAL_PRESENT | {benefit_goal_present:.3f} | {z_goal_norm_avg:.3f}"
        f" | {cal_gap_goal_present:.4f} | {avg_goal_resource_r:.3f}"
        f" | {avg_goal_vs_harm_ratio:.3f} |\n"
        f"| GOAL_ABSENT  | {benefit_goal_absent:.3f} | -- | -- | -- | -- |\n\n"
        f"**Benefit ratio (goal/no-goal): {benefit_ratio:.2f}x**\n\n"
        f"## PASS Criteria\n\n"
        f"| Criterion | Result | Value |\n"
        f"|---|---|---|\n"
        f"| C1: z_goal_norm > 0.1 | {'PASS' if c1_pass else 'FAIL'}"
        f" | {z_goal_norm_avg:.3f} |\n"
        f"| C2: benefit ratio >= 1.3x | {'PASS' if c2_pass else 'FAIL'}"
        f" | {benefit_ratio:.2f}x |\n"
        f"| C3: cal_gap > 0.02 | {'PASS' if c3_pass else 'FAIL'}"
        f" | {cal_gap_goal_present:.4f} |\n"
        f"| C4: no fatal errors | {'PASS' if c4_pass else 'FAIL'} | -- |\n\n"
        f"Criteria met: {criteria_met}/4 -> **{status}**\n\n"
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
        "calibration_gap_goal_present": float(cal_gap_goal_present),
        "goal_resource_r":              float(avg_goal_resource_r),
        "goal_vs_harm_ratio":           float(avg_goal_vs_harm_ratio),
        "avg_resource_seedings":        float(avg_n_resource_seedings),
        "mech124_flag_goal_resource":   float(mech124_flag_goal_resource),
        "mech124_flag_salience":        float(mech124_flag_salience),
        "drive_weight":                 float(drive_weight),
        "novelty_bonus_weight":         float(novelty_bonus_weight),
        "n_seeds":                      float(len(seeds)),
        "alpha_world":                  float(alpha_world),
        "alpha_goal":                   0.3,
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
    parser.add_argument("--warmup",          type=int,   default=500)
    parser.add_argument("--eval-eps",        type=int,   default=100)
    parser.add_argument("--steps",           type=int,   default=200)
    parser.add_argument("--curriculum",      type=int,   default=100)
    parser.add_argument("--alpha-world",     type=float, default=0.9)
    parser.add_argument("--alpha-self",      type=float, default=0.3)
    parser.add_argument("--harm-scale",      type=float, default=0.02)
    parser.add_argument("--proximity-scale", type=float, default=0.05)
    parser.add_argument("--novelty-weight",  type=float, default=0.1)
    parser.add_argument("--drive-weight",    type=float, default=2.0)
    args = parser.parse_args()

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
    )

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
