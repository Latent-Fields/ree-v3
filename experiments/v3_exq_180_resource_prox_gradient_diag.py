#!/opt/local/bin/python3
"""
V3-EXQ-180 -- Resource Proximity Gradient Diagnostic

Claims: MECH-112, SD-012, ARC-030

=== CONTEXT ===

EXQ-179 (H-A/H-B diagnostic) returned:
  goal_tracking_r = -0.056 (H-A CONFIRMED: z_goal is scene noise)
  selection_bias  = 0.0    (H-B CONFIRMED: E3 not using goal score)
  benefit_ratio   = 0.30x  (goal-guided 70% WORSE than random)

The root cause: resources respawn in random locations, so z_goal (seeded from
z_world at contact) encodes a scene snapshot, not a resource-type signal. There
is no fixed direction in z_world space that consistently points toward resources
across respawn events.

=== HYPOTHESIS ===

Instead of a snapshot-based z_goal, use a GRADIENT signal: train
E3's benefit_eval_head (ARC-030) to predict resource proximity
continuously from z_world trajectory states. If z_world encodes
resource proximity at all (even weakly), the head gives E3 a local
gradient to follow -- not "go to scene X" but "prefer states where
resource_prox(z_world) is high."

This also resolves the respawn problem: resource_prox is a CURRENT property
of z_world (how far is the nearest resource right now?), not a stored snapshot
of a past scene.

=== TWO-PART DESIGN ===

PART 1 -- Linear probe (post-warmup):
  After warmup, run 10 probe episodes (random actions). Collect (z_world_t,
  resource_prox_t) pairs. Fit linear regression: is resource_prox linearly
  decodable from z_world at all?
    probe_r2 > 0.05 -> signal exists in z_world
    probe_r2 ~ 0   -> z_world does not encode resource proximity (need encoder supervision)

PART 2 -- benefit_eval_head navigation test:
  In BENEFIT_EVAL_ON condition: train benefit_eval_head with continuous
  resource_prox regression labels during warmup. Enable in E3 scoring. Test
  whether navigation improves.
    benefit_head_r > 0.1 -> head tracks resource proximity in eval (gradient working)
    benefit_ratio > 1.2  -> navigation improved over random baseline

=== PASS CRITERIA ===

C1: probe_r2 > 0.05            (z_world contains resource signal)
C2: benefit_head_r > 0.1       (head tracks resource proximity in eval)
C3: benefit_ratio > 1.2        (navigation improved vs ablated baseline)
C4: n_probe_steps >= 500       (data quality for linear probe)

C1 is a gate: if C1 FAILS, z_world does not encode resource proximity at all.
C2 and C3 failure are then expected and the fix is encoder-level supervision
(resource proximity auxiliary loss in z_world encoder, analogous to SD-009 for harm).

If C1 PASSES and C2 PASSES but C3 FAILS: head is trained but trajectory scoring
is not translating to navigation -- investigate benefit_weight or trajectory diversity.
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


EXPERIMENT_TYPE = "v3_exq_180_resource_prox_gradient_diag"
CLAIM_IDS = ["MECH-112", "SD-012", "ARC-030"]


# ------------------------------------------------------------------ #
# Helpers (from EXQ-179/085h)                                         #
# ------------------------------------------------------------------ #

def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


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


def _compute_probe_r2(
    zw_list: List[torch.Tensor],
    prox_list: List[float],
) -> float:
    """
    R^2 of ridge-regularised linear regression z_world -> resource_prox.
    Returns 0.0 if degenerate or too few samples.
    """
    if len(zw_list) < 8:
        return 0.0
    Z = torch.cat(zw_list, dim=0).float().detach()   # [N, world_dim]
    y = torch.tensor(prox_list, dtype=torch.float32).unsqueeze(1)  # [N, 1]
    ones = torch.ones(Z.shape[0], 1)
    Zb = torch.cat([Z, ones], dim=1)  # [N, world_dim+1]
    lam = 1e-4
    ZtZ = Zb.t() @ Zb + lam * torch.eye(Zb.shape[1])
    Zty = Zb.t() @ y
    try:
        w = torch.linalg.solve(ZtZ, Zty)
        y_pred = Zb @ w
        ss_res = float(((y - y_pred) ** 2).sum().item())
        ss_tot = float(((y - y.mean()) ** 2).sum().item())
        return max(0.0, 1.0 - ss_res / max(ss_tot, 1e-9))
    except Exception:
        return 0.0


# ------------------------------------------------------------------ #
# Main run function                                                    #
# ------------------------------------------------------------------ #

def _run_single(
    seed: int,
    benefit_eval_on: bool,
    warmup_episodes: int,
    probe_episodes: int,
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

    cond_label = "BENEFIT_EVAL_ON" if benefit_eval_on else "BENEFIT_EVAL_OFF"

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
        benefit_eval_enabled=benefit_eval_on,
        benefit_weight=benefit_weight,
    )

    agent = REEAgent(config)

    standard_params = [
        p for n, p in agent.named_parameters()
        if "harm_eval_head" not in n and "benefit_eval_head" not in n
    ]
    harm_eval_params  = list(agent.e3.harm_eval_head.parameters())
    benefit_eval_params = list(agent.e3.benefit_eval_head.parameters())

    optimizer           = optim.Adam(standard_params, lr=lr)
    harm_eval_optimizer = optim.Adam(harm_eval_params, lr=1e-4)
    benefit_eval_optimizer = optim.Adam(benefit_eval_params, lr=1e-4)

    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []
    # Continuous resource_prox regression buffer: (z_world, resource_prox)
    res_buf: List[Tuple[torch.Tensor, float]] = []
    MAX_BUF = 2000

    counts: Dict[str, int] = {
        "hazard_approach": 0, "env_caused_hazard": 0,
        "agent_caused_hazard": 0, "none": 0, "resource": 0,
        "benefit_approach": 0,
    }
    n_benefit_samples_trained = 0

    # --- WARMUP ---
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

            action_idx = random.randint(0, action_dim - 1)
            action_oh  = _action_to_onehot(action_idx, action_dim, agent.device)
            agent._last_action = action_oh

            _, harm_signal, done, info, obs_dict = env.step(action_oh)
            ttype = info.get("transition_type", "none")
            if ttype in counts:
                counts[ttype] += 1

            res_prox_t = _resource_proximity(env)

            # --- Harm buffer ---
            is_harm = float(harm_signal) < 0
            if is_harm:
                harm_buf_pos.append(z_world_curr)
                if len(harm_buf_pos) > MAX_BUF:
                    harm_buf_pos = harm_buf_pos[-MAX_BUF:]
            else:
                harm_buf_neg.append(z_world_curr)
                if len(harm_buf_neg) > MAX_BUF:
                    harm_buf_neg = harm_buf_neg[-MAX_BUF:]

            # --- Resource proximity buffer (BENEFIT_EVAL_ON only) ---
            if benefit_eval_on:
                res_buf.append((z_world_curr, res_prox_t))
                if len(res_buf) > MAX_BUF:
                    res_buf = res_buf[-MAX_BUF:]

            # --- E1 + E2 training ---
            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total_loss = e1_loss + e2_loss
            if total_loss.requires_grad:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

            # --- Harm eval training ---
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
                harm_loss = F.mse_loss(agent.e3.harm_eval(zw_b), target)
                if harm_loss.requires_grad:
                    harm_eval_optimizer.zero_grad()
                    harm_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        agent.e3.harm_eval_head.parameters(), 0.5,
                    )
                    harm_eval_optimizer.step()

            # --- benefit_eval_head training: resource_prox regression ---
            if benefit_eval_on and len(res_buf) >= 8:
                k = min(32, len(res_buf))
                batch = random.sample(res_buf, k)
                zw_b = torch.cat([b[0] for b in batch], dim=0)
                prox_b = torch.tensor(
                    [[b[1]] for b in batch],
                    dtype=torch.float32,
                    device=agent.device,
                )
                b_pred = agent.e3.benefit_eval(zw_b)
                b_loss = F.mse_loss(b_pred, prox_b)
                if b_loss.requires_grad:
                    benefit_eval_optimizer.zero_grad()
                    b_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        agent.e3.benefit_eval_head.parameters(), 0.5,
                    )
                    benefit_eval_optimizer.step()
                agent.e3.record_benefit_sample(1)
                n_benefit_samples_trained += 1

            if done:
                break

        if (ep + 1) % 100 == 0 or ep == warmup_episodes - 1:
            curriculum_tag = " [curriculum]" if ep < curriculum_episodes else ""
            print(
                f"  [train] seed={seed} cond={cond_label}"
                f" ep {ep+1}/{warmup_episodes}"
                f" resource={counts['resource']}"
                f" res_buf={len(res_buf)}"
                f" benefit_trained={n_benefit_samples_trained}"
                f"{curriculum_tag}",
                flush=True,
            )

    # --- PART 1: Linear probe (post-warmup, BENEFIT_EVAL_ON only) ---
    probe_r2 = 0.0
    benefit_r2_probe = 0.0  # head predictions vs resource_prox on probe data
    n_probe_steps = 0
    agent.eval()

    if benefit_eval_on:
        probe_zw: List[torch.Tensor] = []
        probe_prox: List[float] = []
        probe_head_pred: List[float] = []

        for _ in range(probe_episodes):
            _, obs_dict = env.reset()
            agent.reset()
            for _ in range(steps_per_episode):
                obs_body  = obs_dict["body_state"]
                obs_world = obs_dict["world_state"]
                obs_harm  = obs_dict.get("harm_obs", None)

                with torch.no_grad():
                    latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
                agent.clock.advance()

                z_world_t = latent.z_world.detach()
                rp_t = _resource_proximity(env)
                probe_zw.append(z_world_t)
                probe_prox.append(rp_t)
                n_probe_steps += 1

                with torch.no_grad():
                    head_pred = agent.e3.benefit_eval(z_world_t).mean().item()
                probe_head_pred.append(head_pred)

                action_idx = random.randint(0, action_dim - 1)
                action_oh  = _action_to_onehot(action_idx, action_dim, agent.device)
                agent._last_action = action_oh
                _, _, done, _, obs_dict = env.step(action_oh)
                if done:
                    break

        probe_r2 = _compute_probe_r2(probe_zw, probe_prox)
        # Head's predictions on probe data vs true resource_prox (squared Pearson r)
        head_r_probe = _pearson_r(probe_head_pred, probe_prox)
        benefit_r2_probe = head_r_probe ** 2

        print(
            f"  [probe] seed={seed} cond={cond_label}"
            f" n_steps={n_probe_steps}"
            f" probe_r2={probe_r2:.4f}"
            f" benefit_r2_probe={benefit_r2_probe:.4f}"
            f" head_r_probe={head_r_probe:.3f}",
            flush=True,
        )

    # --- EVAL ---
    benefit_per_ep: List[float] = []
    harm_buf_eval_pos: List[torch.Tensor] = []
    harm_buf_eval_neg: List[torch.Tensor] = []
    n_eval_steps = 0

    # Gradient diagnostic: benefit_eval(z_world_t) vs resource_prox_t
    benefit_head_t_vals: List[float] = []
    resource_prox_t_vals: List[float] = []

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

            ticks = agent.clock.advance()
            z_world_curr = latent.z_world.detach()

            # Gradient tracking: measure head prediction vs ground truth every step
            if benefit_eval_on:
                with torch.no_grad():
                    head_t = agent.e3.benefit_eval(z_world_curr).mean().item()
                rp_t = _resource_proximity(env)
                benefit_head_t_vals.append(head_t)
                resource_prox_t_vals.append(rp_t)
                n_eval_steps += 1

            # Action selection
            if benefit_eval_on and ticks.get("e3_tick", True):
                with torch.no_grad():
                    e1_prior = agent._e1_tick(latent)
                    candidates = agent._e3_tick(latent, e1_prior)
                    result = agent.e3.select(candidates, temperature=temperature)
                action_oh = result.selected_action
                agent._last_action = action_oh
            elif benefit_eval_on:
                # Hold action between E3 ticks
                if agent._last_action is None:
                    action_oh = _action_to_onehot(
                        random.randint(0, action_dim - 1), action_dim, agent.device
                    )
                    agent._last_action = action_oh
                else:
                    action_oh = agent._last_action
            else:
                # BENEFIT_EVAL_OFF: random actions
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

    calibration_gap = 0.0
    if len(harm_buf_eval_pos) >= 4 and len(harm_buf_eval_neg) >= 4:
        k = min(64, min(len(harm_buf_eval_pos), len(harm_buf_eval_neg)))
        zw_pos = torch.cat(harm_buf_eval_pos[-k:], dim=0)
        zw_neg = torch.cat(harm_buf_eval_neg[-k:], dim=0)
        with torch.no_grad():
            harm_pos = agent.e3.harm_eval(zw_pos).mean().item()
            harm_neg = agent.e3.harm_eval(zw_neg).mean().item()
        calibration_gap = harm_pos - harm_neg

    # Key gradient metric: does benefit_eval_head track resource proximity?
    benefit_head_r = (
        _pearson_r(benefit_head_t_vals, resource_prox_t_vals)
        if benefit_eval_on and len(benefit_head_t_vals) >= 4 else 0.0
    )

    print(
        f"  [eval] seed={seed} cond={cond_label}"
        f" avg_benefit/ep={avg_benefit:.3f}"
        f" cal_gap={calibration_gap:.4f}"
        f" probe_r2={probe_r2:.4f}"
        f" benefit_head_r={benefit_head_r:.3f}"
        f" n_eval_steps={n_eval_steps}",
        flush=True,
    )

    return {
        "seed": seed,
        "condition": cond_label,
        "benefit_eval_on": benefit_eval_on,
        "avg_benefit_per_ep": float(avg_benefit),
        "probe_r2": float(probe_r2),
        "benefit_r2_probe": float(benefit_r2_probe),
        "benefit_head_r": float(benefit_head_r),
        "n_probe_steps": int(n_probe_steps),
        "n_eval_steps": int(n_eval_steps),
        "calibration_gap": float(calibration_gap),
        "n_benefit_samples_trained": int(n_benefit_samples_trained),
        "train_resource_events": int(counts["resource"]),
    }


def run(
    seeds: Tuple = (42, 7),
    warmup_episodes: int = 400,
    probe_episodes: int = 10,
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
    benefit_weight: float = 1.0,
    temperature: float = 1.0,
    **kwargs,
) -> dict:
    """BENEFIT_EVAL_ON (E3 resource_prox head) vs BENEFIT_EVAL_OFF (random)."""
    results_on:  List[Dict] = []
    results_off: List[Dict] = []

    for seed in seeds:
        for benefit_eval_on in [True, False]:
            label = "BENEFIT_EVAL_ON" if benefit_eval_on else "BENEFIT_EVAL_OFF"
            print(
                f"\n[V3-EXQ-180] {label} seed={seed}"
                f" warmup={warmup_episodes} probe={probe_episodes} eval={eval_episodes}"
                f" alpha_world={alpha_world} benefit_weight={benefit_weight}",
                flush=True,
            )
            r = _run_single(
                seed=seed,
                benefit_eval_on=benefit_eval_on,
                warmup_episodes=warmup_episodes,
                probe_episodes=probe_episodes,
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
            if benefit_eval_on:
                results_on.append(r)
            else:
                results_off.append(r)

    def _avg(results: List[Dict], key: str) -> float:
        vals = [r[key] for r in results]
        return float(sum(vals) / max(1, len(vals)))

    avg_probe_r2        = _avg(results_on, "probe_r2")
    avg_benefit_r2      = _avg(results_on, "benefit_r2_probe")
    avg_benefit_head_r  = _avg(results_on, "benefit_head_r")
    avg_n_probe_steps   = _avg(results_on, "n_probe_steps")
    avg_benefit_trained = _avg(results_on, "n_benefit_samples_trained")
    avg_cal_gap         = _avg(results_on, "calibration_gap")
    benefit_on          = _avg(results_on, "avg_benefit_per_ep")
    benefit_off         = _avg(results_off, "avg_benefit_per_ep")

    benefit_ratio = (
        benefit_on / max(1e-6, benefit_off)
        if benefit_off > 1e-6 else 0.0
    )

    c1_pass = avg_probe_r2 > 0.05
    c2_pass = avg_benefit_head_r > 0.1
    c3_pass = benefit_ratio > 1.2
    c4_pass = avg_n_probe_steps >= 500

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass])
    status = "PASS" if all_pass else "FAIL"

    # --- Interpretation ---
    if not c1_pass:
        c1_interp = (
            f"C1 FAIL: probe_r2={avg_probe_r2:.4f} <= 0.05."
            " z_world does NOT encode resource proximity."
            " Fix: add resource_prox auxiliary loss to z_world encoder (SD-009 analog)."
        )
    else:
        c1_interp = (
            f"C1 PASS: probe_r2={avg_probe_r2:.4f} > 0.05."
            " z_world DOES contain resource proximity signal."
        )

    if not c2_pass:
        if c1_pass:
            c2_interp = (
                f"C2 FAIL: benefit_head_r={avg_benefit_head_r:.3f} <= 0.1."
                " Head did not decode the available signal."
                " Check: n_benefit_samples_trained, lr, or training label quality."
            )
        else:
            c2_interp = (
                f"C2 FAIL: benefit_head_r={avg_benefit_head_r:.3f} <= 0.1."
                " Expected -- no signal in z_world to decode (C1 failed)."
            )
    else:
        c2_interp = (
            f"C2 PASS: benefit_head_r={avg_benefit_head_r:.3f} > 0.1."
            " Head tracks resource proximity in eval."
        )

    if not c3_pass:
        if c2_pass:
            c3_interp = (
                f"C3 FAIL: benefit_ratio={benefit_ratio:.2f}x <= 1.2."
                " Head tracks proximity but E3 scoring does not improve navigation."
                " Consider increasing benefit_weight or checking trajectory diversity."
            )
        else:
            c3_interp = (
                f"C3 FAIL: benefit_ratio={benefit_ratio:.2f}x <= 1.2."
                " Expected -- head not tracking proximity."
            )
    else:
        c3_interp = (
            f"C3 PASS: benefit_ratio={benefit_ratio:.2f}x > 1.2."
            " benefit_eval_head improves resource navigation."
        )

    print(f"\n[V3-EXQ-180] Final results:", flush=True)
    print(
        f"  probe_r2={avg_probe_r2:.4f}"
        f"  benefit_r2_probe={avg_benefit_r2:.4f}"
        f"  benefit_head_r={avg_benefit_head_r:.3f}",
        flush=True,
    )
    print(
        f"  benefit_on={benefit_on:.3f}"
        f"  benefit_off={benefit_off:.3f}"
        f"  ratio={benefit_ratio:.2f}x",
        flush=True,
    )
    print(f"  {c1_interp}", flush=True)
    print(f"  {c2_interp}", flush=True)
    print(f"  {c3_interp}", flush=True)
    print(f"  status={status} ({criteria_met}/4)", flush=True)

    per_on_rows = "\n".join(
        f"  seed={r['seed']}:"
        f" benefit/ep={r['avg_benefit_per_ep']:.3f}"
        f" probe_r2={r['probe_r2']:.4f}"
        f" benefit_head_r={r['benefit_head_r']:.3f}"
        f" n_trained={r['n_benefit_samples_trained']}"
        for r in results_on
    )
    per_off_rows = "\n".join(
        f"  seed={r['seed']}: benefit/ep={r['avg_benefit_per_ep']:.3f}"
        for r in results_off
    )

    failure_notes: List[str] = []
    if not c1_pass:
        failure_notes.append(c1_interp)
    if not c2_pass and c1_pass:
        failure_notes.append(c2_interp)
    if not c3_pass and c2_pass:
        failure_notes.append(c3_interp)
    if not c4_pass:
        failure_notes.append(
            f"C4 FAIL: n_probe_steps={avg_n_probe_steps:.0f} < 500."
            " Increase probe_episodes."
        )

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(
            f"- {n}" for n in failure_notes
        )

    summary_markdown = (
        f"# V3-EXQ-180 -- Resource Proximity Gradient Diagnostic\n\n"
        f"**Status:** {status}\n"
        f"**Claims:** MECH-112, SD-012, ARC-030\n"
        f"**Seeds:** {list(seeds)}\n\n"
        f"## Context\n\n"
        f"EXQ-179 confirmed: goal_tracking_r=-0.056 (H-A), selection_bias~0 (H-B)."
        f" z_goal seeding from z_world is scene noise, not a resource gradient."
        f" This experiment tests whether benefit_eval_head (ARC-030) can provide"
        f" the missing gradient by learning resource_prox from z_world directly.\n\n"
        f"## Two-Part Design\n\n"
        f"**Part 1 (linear probe):** Is resource proximity linearly decodable from z_world?\n"
        f"  probe_r2 = R^2 of ridge regression z_world -> resource_prox"
        f" on {probe_episodes} random-action probe episodes post-warmup.\n\n"
        f"**Part 2 (navigation test):** Does training benefit_eval_head on resource_prox"
        f" labels improve E3 trajectory selection?\n"
        f"  benefit_head_r = Pearson r(benefit_eval(z_world_t), resource_prox_t) in eval\n"
        f"  benefit_ratio = benefit_EVAL_ON / benefit_EVAL_OFF\n\n"
        f"**alpha_world:** {alpha_world}  (SD-008)\n"
        f"**benefit_weight:** {benefit_weight}\n"
        f"**Warmup:** {warmup_episodes} eps (curriculum={curriculum_episodes} eps)\n"
        f"**Probe:** {probe_episodes} eps (random actions, post-warmup)\n"
        f"**Eval:** {eval_episodes} eps per condition\n\n"
        f"## Key Results\n\n"
        f"| Metric | Value | Threshold | Interpretation |\n"
        f"|---|---|---|---|\n"
        f"| probe_r2 (Part 1) | {avg_probe_r2:.4f} | > 0.05 = signal in z_world | {c1_interp[:60]}... |\n"
        f"| benefit_head_r | {avg_benefit_head_r:.3f} | > 0.1 = head tracks gradient | {c2_interp[:60]}... |\n"
        f"| benefit_ratio | {benefit_ratio:.2f}x | > 1.2 = navigation improved | {c3_interp[:60]}... |\n\n"
        f"## Navigation Results\n\n"
        f"| Condition | benefit/ep | probe_r2 | benefit_head_r |\n"
        f"|---|---|---|---|\n"
        f"| BENEFIT_EVAL_ON | {benefit_on:.3f} | {avg_probe_r2:.4f} | {avg_benefit_head_r:.3f} |\n"
        f"| BENEFIT_EVAL_OFF (random) | {benefit_off:.3f} | -- | -- |\n\n"
        f"**Benefit ratio: {benefit_ratio:.2f}x**\n\n"
        f"## PASS Criteria\n\n"
        f"| Criterion | Result | Value |\n"
        f"|---|---|---|\n"
        f"| C1: probe_r2 > 0.05 | {'PASS' if c1_pass else 'FAIL'} | {avg_probe_r2:.4f} |\n"
        f"| C2: benefit_head_r > 0.1 | {'PASS' if c2_pass else 'FAIL'} | {avg_benefit_head_r:.3f} |\n"
        f"| C3: benefit_ratio > 1.2 | {'PASS' if c3_pass else 'FAIL'} | {benefit_ratio:.2f}x |\n"
        f"| C4: n_probe_steps >= 500 | {'PASS' if c4_pass else 'FAIL'} | {avg_n_probe_steps:.0f} |\n\n"
        f"Criteria met: {criteria_met}/4 -> **{status}**\n\n"
        f"## Interpretation\n\n"
        f"{c1_interp}\n\n{c2_interp}\n\n{c3_interp}\n\n"
        f"**benefit_r2_probe (head fit on probe data):** {avg_benefit_r2:.4f}\n\n"
        f"## Per-Seed\n\n"
        f"BENEFIT_EVAL_ON:\n{per_on_rows}\n\n"
        f"BENEFIT_EVAL_OFF:\n{per_off_rows}\n"
        f"{failure_section}\n"
    )

    evidence_direction = (
        "supports" if all_pass
        else ("mixed" if criteria_met >= 2 else "weakens")
    )

    metrics = {
        "probe_r2":                 float(avg_probe_r2),
        "benefit_r2_probe":         float(avg_benefit_r2),
        "benefit_head_r":           float(avg_benefit_head_r),
        "benefit_per_ep_on":        float(benefit_on),
        "benefit_per_ep_off":       float(benefit_off),
        "benefit_ratio":            float(benefit_ratio),
        "calibration_gap_on":       float(avg_cal_gap),
        "n_probe_steps_avg":        float(avg_n_probe_steps),
        "n_benefit_samples_avg":    float(avg_benefit_trained),
        "crit1_pass":               1.0 if c1_pass else 0.0,
        "crit2_pass":               1.0 if c2_pass else 0.0,
        "crit3_pass":               1.0 if c3_pass else 0.0,
        "crit4_pass":               1.0 if c4_pass else 0.0,
        "criteria_met":             float(criteria_met),
        "n_seeds":                  float(len(seeds)),
        "alpha_world":              float(alpha_world),
        "benefit_weight":           float(benefit_weight),
    }

    return {
        "status": status,
        "metrics": metrics,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": evidence_direction,
        "experiment_type": EXPERIMENT_TYPE,
        "fatal_error_count": 0,
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",            type=int,   nargs="+", default=[42, 7])
    parser.add_argument("--warmup",           type=int,   default=400)
    parser.add_argument("--probe-eps",        type=int,   default=10)
    parser.add_argument("--eval-eps",         type=int,   default=60)
    parser.add_argument("--steps",            type=int,   default=200)
    parser.add_argument("--curriculum",       type=int,   default=80)
    parser.add_argument("--alpha-world",      type=float, default=0.9)
    parser.add_argument("--alpha-self",       type=float, default=0.3)
    parser.add_argument("--harm-scale",       type=float, default=0.02)
    parser.add_argument("--proximity-scale",  type=float, default=0.05)
    parser.add_argument("--novelty-weight",   type=float, default=0.1)
    parser.add_argument("--benefit-weight",   type=float, default=1.0)
    parser.add_argument("--temperature",      type=float, default=1.0)
    parser.add_argument("--dry-run",          action="store_true",
                        help="Run 1 seed, 5 warmup, 3 probe, 5 eval for smoke test")
    args = parser.parse_args()

    if args.dry_run:
        args.seeds     = [42]
        args.warmup    = 5
        args.probe_eps = 3
        args.eval_eps  = 5
        args.curriculum = 2
        print("[DRY-RUN] 1 seed, 5 warmup, 3 probe, 5 eval", flush=True)

    result = run(
        seeds=tuple(args.seeds),
        warmup_episodes=args.warmup,
        probe_episodes=args.probe_eps,
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
