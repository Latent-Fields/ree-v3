#!/opt/local/bin/python3
"""
V3-EXQ-180a -- Resource Proximity benefit_eval_head with Phased Training (ARC-030)

Supersedes: V3-EXQ-180

=== CONTEXT ===

EXQ-180 showed probe_r2=0.619: z_world DOES encode resource proximity (C1 PASS).
But benefit_ratio=0.29x (C3 FAIL): the benefit_eval_head trained jointly with E1+E2
could not exploit the signal. Root cause: joint training -- benefit_eval_head chased
moving z_world during E1 convergence (same pattern as EXQ-166b/c/d identity collapse,
EXQ-085l position noise, EXQ-194 w_goal constant).

=== FIX: PHASED TRAINING ===

Phase 0 (200 eps): E1+E2 warmup ONLY. No benefit_eval training. Let z_world converge
    to a stable representation.

Phase 1 (100 eps): Freeze E1+E2. Train benefit_eval_head on z_world.detach() with
    MSE on resource_proximity labels (continuous, from resource_field_view.max()).
    Also train a linear probe for diagnostic R^2 comparison.

Phase 2 (200 eps): Eval with benefit_eval_enabled=True in E3 trajectory scoring.
    BENEFIT_EVAL_ON condition uses E3 selection; BENEFIT_EVAL_OFF uses random actions.

=== RESOURCE PROXIMITY LABELS ===

Using resource_field_view from obs_dict (continuous gradient field, not binary contact).
resource_field_view is a 5x5 normalised proximity field. .max() gives the peak proximity
in the agent's local view -- continuous in [0, 1], strongest when a resource is adjacent.

This is better than the manhattan-distance proxy used in EXQ-180 because:
  (1) It's what z_world actually encodes (field is part of world_state obs)
  (2) It's continuous and differentiable-friendly for MSE training
  (3) No need for _resource_proximity() helper that queries env internals

=== PASS CRITERIA ===

C1: probe_r2 > 0.3       (z_world encodes resource proximity -- baseline was 0.619)
C2: benefit_head_r > 0.3  (head tracks proximity after phased training)
C3: benefit_ratio > 1.0   (net positive navigation effect)

PASS = all three criteria pass in >= 2/3 seeds.

Claims: ARC-030
"""

import sys
import random
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_180a_arc030_benefit_eval_phased"
CLAIM_IDS = ["ARC-030"]


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


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
    """R^2 of ridge-regularised linear regression z_world -> resource_prox."""
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
# Single-seed run with phased training                                 #
# ------------------------------------------------------------------ #

def _run_single(
    seed: int,
    p0_episodes: int,
    p1_episodes: int,
    p2_episodes: int,
    steps_per_episode: int,
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
    num_hazards: int,
    num_resources: int,
) -> Dict:
    torch.manual_seed(seed)
    random.seed(seed)

    env = CausalGridWorldV2(
        seed=seed,
        size=10,
        num_hazards=num_hazards,
        num_resources=num_resources,
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

    # benefit_eval_enabled=False during P0/P1; we manually enable for P2 eval
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
        benefit_eval_enabled=False,   # Off during training phases
        benefit_weight=benefit_weight,
    )

    agent = REEAgent(config)

    # Separate parameter groups
    standard_params = [
        p for n, p in agent.named_parameters()
        if "harm_eval_head" not in n and "benefit_eval_head" not in n
    ]
    harm_eval_params = list(agent.e3.harm_eval_head.parameters())
    benefit_eval_params = list(agent.e3.benefit_eval_head.parameters())

    # Optimizers: standard for P0, benefit for P1
    agent_optimizer = optim.Adam(standard_params, lr=lr)
    harm_eval_optimizer = optim.Adam(harm_eval_params, lr=1e-4)
    benefit_eval_optimizer = optim.Adam(benefit_eval_params, lr=1e-4)

    # Harm buffers for harm_eval training during P0
    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []
    MAX_BUF = 2000

    # ================================================================ #
    # PHASE 0: E1+E2 warmup (+ harm_eval). NO benefit_eval training.   #
    # ================================================================ #
    print(f"\n  [P0] seed={seed} E1+E2 warmup ({p0_episodes} eps)", flush=True)
    agent.train()

    for ep in range(p0_episodes):
        _, obs_dict = env.reset()
        agent.reset()

        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_harm = obs_dict.get("harm_obs", None)

            latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
            agent.clock.advance()
            z_world_curr = latent.z_world.detach()

            action_idx = random.randint(0, action_dim - 1)
            action_oh = _action_to_onehot(action_idx, action_dim, agent.device)
            agent._last_action = action_oh

            _, harm_signal, done, info, obs_dict = env.step(action_oh)

            # Harm buffer
            if float(harm_signal) < 0:
                harm_buf_pos.append(z_world_curr)
                if len(harm_buf_pos) > MAX_BUF:
                    harm_buf_pos = harm_buf_pos[-MAX_BUF:]
            else:
                harm_buf_neg.append(z_world_curr)
                if len(harm_buf_neg) > MAX_BUF:
                    harm_buf_neg = harm_buf_neg[-MAX_BUF:]

            # E1 + E2 training
            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total_loss = e1_loss + e2_loss
            if total_loss.requires_grad:
                agent_optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(standard_params, 1.0)
                agent_optimizer.step()

            # Harm eval training (separate optimizer, on detached z_world)
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
                harm_loss = F.mse_loss(agent.e3.harm_eval(zw_b), target)
                if harm_loss.requires_grad:
                    harm_eval_optimizer.zero_grad()
                    harm_loss.backward()
                    torch.nn.utils.clip_grad_norm_(harm_eval_params, 0.5)
                    harm_eval_optimizer.step()

            if done:
                break

        if (ep + 1) % 50 == 0 or ep == p0_episodes - 1:
            print(
                f"    P0 ep {ep+1}/{p0_episodes}"
                f" harm_pos={len(harm_buf_pos)} harm_neg={len(harm_buf_neg)}",
                flush=True,
            )

    # ================================================================ #
    # PHASE 1: Freeze E1+E2. Train benefit_eval_head on detached z_world. #
    # Also fit linear probe for diagnostic.                              #
    # ================================================================ #
    print(f"\n  [P1] seed={seed} benefit_eval_head training ({p1_episodes} eps)", flush=True)
    agent.eval()  # Freeze encoder

    # Collect (z_world, resource_prox) pairs for benefit_eval training
    benefit_loss_history: List[float] = []
    probe_zw_all: List[torch.Tensor] = []
    probe_prox_all: List[float] = []

    for ep in range(p1_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        ep_benefit_loss = 0.0
        ep_steps = 0

        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_harm = obs_dict.get("harm_obs", None)

            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
            agent.clock.advance()

            # Stable z_world from frozen encoder
            z_world_stable = latent.z_world.detach()

            # Resource proximity label from resource_field_view
            resource_field = obs_dict.get("resource_field_view", None)
            if resource_field is not None:
                res_prox = float(resource_field.max().item())
            else:
                res_prox = 0.0

            # Collect for probe
            probe_zw_all.append(z_world_stable)
            probe_prox_all.append(res_prox)

            # Train benefit_eval_head on this sample (online SGD with mini-batches)
            # Accumulate a small batch before updating
            if len(probe_zw_all) >= 8 and ep_steps % 4 == 0:
                k = min(32, len(probe_zw_all))
                indices = random.sample(range(len(probe_zw_all)), k)
                zw_batch = torch.cat([probe_zw_all[i] for i in indices], dim=0)
                prox_batch = torch.tensor(
                    [[probe_prox_all[i]] for i in indices],
                    dtype=torch.float32,
                    device=agent.device,
                )
                b_pred = agent.e3.benefit_eval(zw_batch)
                b_loss = F.mse_loss(b_pred, prox_batch)
                if b_loss.requires_grad:
                    benefit_eval_optimizer.zero_grad()
                    b_loss.backward()
                    torch.nn.utils.clip_grad_norm_(benefit_eval_params, 0.5)
                    benefit_eval_optimizer.step()
                ep_benefit_loss += b_loss.item()
                ep_steps += 1
                agent.e3.record_benefit_sample(1)

            action_idx = random.randint(0, action_dim - 1)
            action_oh = _action_to_onehot(action_idx, action_dim, agent.device)
            agent._last_action = action_oh
            _, _, done, _, obs_dict = env.step(action_oh)
            if done:
                break

        avg_ep_loss = ep_benefit_loss / max(1, ep_steps)
        benefit_loss_history.append(avg_ep_loss)

        if (ep + 1) % 25 == 0 or ep == p1_episodes - 1:
            print(
                f"    P1 ep {ep+1}/{p1_episodes}"
                f" benefit_loss={avg_ep_loss:.6f}"
                f" n_samples={len(probe_zw_all)}",
                flush=True,
            )

    # Compute linear probe R^2 on all P1 data
    probe_r2 = _compute_probe_r2(probe_zw_all, probe_prox_all)

    # Also compute benefit_eval_head predictions on P1 data for diagnostic
    head_preds_p1: List[float] = []
    with torch.no_grad():
        for zw in probe_zw_all:
            pred = agent.e3.benefit_eval(zw).mean().item()
            head_preds_p1.append(pred)
    head_r_p1 = _pearson_r(head_preds_p1, probe_prox_all)

    final_benefit_loss = benefit_loss_history[-1] if benefit_loss_history else 999.0
    n_p1_samples = len(probe_zw_all)

    print(
        f"  [P1 done] seed={seed}"
        f" probe_r2={probe_r2:.4f}"
        f" head_r_p1={head_r_p1:.4f}"
        f" final_loss={final_benefit_loss:.6f}"
        f" n_samples={n_p1_samples}",
        flush=True,
    )

    # ================================================================ #
    # PHASE 2: Eval -- benefit_eval_enabled in E3 scoring               #
    # ================================================================ #
    print(f"\n  [P2] seed={seed} Evaluation ({p2_episodes} eps)", flush=True)

    # Enable benefit_eval in E3 scoring
    agent.config.benefit_eval_enabled = True
    agent.e3.config.benefit_eval_enabled = True
    agent.eval()

    # Run two conditions: BENEFIT_EVAL_ON (E3 selection) and BENEFIT_EVAL_OFF (random)
    results_by_cond: Dict[str, Dict] = {}

    for cond_label, use_e3 in [("BENEFIT_EVAL_ON", True), ("BENEFIT_EVAL_OFF", False)]:
        benefit_per_ep: List[float] = []
        head_preds_eval: List[float] = []
        prox_vals_eval: List[float] = []

        # Disable benefit_eval for OFF condition
        if not use_e3:
            agent.config.benefit_eval_enabled = False
            agent.e3.config.benefit_eval_enabled = False

        for _ in range(p2_episodes):
            _, obs_dict = env.reset()
            agent.reset()
            ep_benefit = 0.0

            for _ in range(steps_per_episode):
                obs_body = obs_dict["body_state"]
                obs_world = obs_dict["world_state"]
                obs_harm = obs_dict.get("harm_obs", None)

                with torch.no_grad():
                    latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
                ticks = agent.clock.advance()
                z_world_curr = latent.z_world.detach()

                # Track head predictions vs true proximity
                if use_e3:
                    with torch.no_grad():
                        head_pred = agent.e3.benefit_eval(z_world_curr).mean().item()
                    resource_field = obs_dict.get("resource_field_view", None)
                    rp = float(resource_field.max().item()) if resource_field is not None else 0.0
                    head_preds_eval.append(head_pred)
                    prox_vals_eval.append(rp)

                # Action selection
                if use_e3 and ticks.get("e3_tick", True):
                    with torch.no_grad():
                        e1_prior = agent._e1_tick(latent)
                        candidates = agent._e3_tick(latent, e1_prior)
                        result = agent.e3.select(candidates, temperature=temperature)
                    action_oh = result.selected_action
                    agent._last_action = action_oh
                elif use_e3:
                    if agent._last_action is None:
                        action_oh = _action_to_onehot(
                            random.randint(0, action_dim - 1), action_dim, agent.device
                        )
                        agent._last_action = action_oh
                    else:
                        action_oh = agent._last_action
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

                if done:
                    break

            benefit_per_ep.append(ep_benefit)

        avg_benefit = float(sum(benefit_per_ep) / max(1, len(benefit_per_ep)))
        benefit_head_r = (
            _pearson_r(head_preds_eval, prox_vals_eval)
            if use_e3 and len(head_preds_eval) >= 4 else 0.0
        )

        results_by_cond[cond_label] = {
            "avg_benefit_per_ep": avg_benefit,
            "benefit_head_r": benefit_head_r,
            "n_eval_steps": len(head_preds_eval) if use_e3 else 0,
        }

        print(
            f"    P2 {cond_label}: avg_benefit/ep={avg_benefit:.3f}"
            f" head_r={benefit_head_r:.4f}",
            flush=True,
        )

        # Re-enable for next condition check
        agent.config.benefit_eval_enabled = True
        agent.e3.config.benefit_eval_enabled = True

    benefit_on = results_by_cond["BENEFIT_EVAL_ON"]["avg_benefit_per_ep"]
    benefit_off = results_by_cond["BENEFIT_EVAL_OFF"]["avg_benefit_per_ep"]
    benefit_head_r = results_by_cond["BENEFIT_EVAL_ON"]["benefit_head_r"]
    benefit_ratio = benefit_on / max(1e-6, benefit_off) if benefit_off > 1e-6 else 0.0

    # Per-seed criteria
    c1_pass = 1 if probe_r2 > 0.3 else 0
    c2_pass = 1 if benefit_head_r > 0.3 else 0
    c3_pass = 1 if benefit_ratio > 1.0 else 0

    print(
        f"  [P2 done] seed={seed}"
        f" probe_r2={probe_r2:.4f} (C1={'PASS' if c1_pass else 'FAIL'})"
        f" head_r={benefit_head_r:.4f} (C2={'PASS' if c2_pass else 'FAIL'})"
        f" ratio={benefit_ratio:.2f}x (C3={'PASS' if c3_pass else 'FAIL'})",
        flush=True,
    )

    return {
        "seed": seed,
        "probe_r2": float(probe_r2),
        "benefit_head_r": float(benefit_head_r),
        "head_r_p1": float(head_r_p1),
        "benefit_per_ep_on": float(benefit_on),
        "benefit_per_ep_off": float(benefit_off),
        "benefit_ratio": float(benefit_ratio),
        "final_benefit_loss": float(final_benefit_loss),
        "n_p1_samples": int(n_p1_samples),
        "c1_pass": c1_pass,
        "c2_pass": c2_pass,
        "c3_pass": c3_pass,
    }


# ------------------------------------------------------------------ #
# Main entry                                                           #
# ------------------------------------------------------------------ #

def run(
    seeds: Tuple = (42, 7, 11),
    p0_episodes: int = 200,
    p1_episodes: int = 100,
    p2_episodes: int = 200,
    steps_per_episode: int = 200,
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
    num_hazards: int = 4,
    num_resources: int = 3,
    **kwargs,
) -> dict:
    """Phased training of benefit_eval_head on resource proximity."""
    all_results: List[Dict] = []

    for seed in seeds:
        print(
            f"\n{'='*60}\n"
            f"[V3-EXQ-180a] seed={seed}"
            f" P0={p0_episodes} P1={p1_episodes} P2={p2_episodes}"
            f" alpha_world={alpha_world} benefit_weight={benefit_weight}"
            f" hazards={num_hazards} resources={num_resources}\n"
            f"{'='*60}",
            flush=True,
        )
        r = _run_single(
            seed=seed,
            p0_episodes=p0_episodes,
            p1_episodes=p1_episodes,
            p2_episodes=p2_episodes,
            steps_per_episode=steps_per_episode,
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
            num_hazards=num_hazards,
            num_resources=num_resources,
        )
        all_results.append(r)

    # Aggregate
    def _avg(key: str) -> float:
        return float(sum(r[key] for r in all_results) / len(all_results))

    avg_probe_r2 = _avg("probe_r2")
    avg_head_r = _avg("benefit_head_r")
    avg_head_r_p1 = _avg("head_r_p1")
    avg_benefit_on = _avg("benefit_per_ep_on")
    avg_benefit_off = _avg("benefit_per_ep_off")
    avg_ratio = avg_benefit_on / max(1e-6, avg_benefit_off) if avg_benefit_off > 1e-6 else 0.0
    avg_loss = _avg("final_benefit_loss")

    seeds_c1 = sum(r["c1_pass"] for r in all_results)
    seeds_c2 = sum(r["c2_pass"] for r in all_results)
    seeds_c3 = sum(r["c3_pass"] for r in all_results)
    n_seeds = len(seeds)
    pass_threshold = (n_seeds + 1) // 2 + (1 if n_seeds % 2 == 0 else 0)  # majority: 2/3

    c1_overall = seeds_c1 >= pass_threshold
    c2_overall = seeds_c2 >= pass_threshold
    c3_overall = seeds_c3 >= pass_threshold
    all_pass = c1_overall and c2_overall and c3_overall
    status = "PASS" if all_pass else "FAIL"

    # Interpretation
    c1_interp = (
        f"C1 {'PASS' if c1_overall else 'FAIL'}: probe_r2={avg_probe_r2:.4f}"
        f" ({'>' if c1_overall else '<='} 0.3)"
        f" ({seeds_c1}/{n_seeds} seeds)."
        f" z_world {'DOES' if c1_overall else 'does NOT'} encode resource proximity."
    )
    c2_interp = (
        f"C2 {'PASS' if c2_overall else 'FAIL'}: benefit_head_r={avg_head_r:.4f}"
        f" ({'>' if c2_overall else '<='} 0.3)"
        f" ({seeds_c2}/{n_seeds} seeds)."
    )
    if c2_overall:
        c2_interp += " Phased-trained head tracks resource proximity."
    elif c1_overall:
        c2_interp += (
            " Signal exists in z_world but head did not learn it."
            " Check lr, n_p1_samples, or head capacity."
        )
    else:
        c2_interp += " Expected -- no signal in z_world."

    c3_interp = (
        f"C3 {'PASS' if c3_overall else 'FAIL'}: benefit_ratio={avg_ratio:.2f}x"
        f" ({'>' if c3_overall else '<='} 1.0)"
        f" ({seeds_c3}/{n_seeds} seeds)."
    )
    if c3_overall:
        c3_interp += " benefit_eval_head improves resource navigation."
    elif c2_overall:
        c3_interp += (
            " Head tracks proximity but E3 scoring does not improve navigation."
            " Check benefit_weight or trajectory diversity."
        )
    else:
        c3_interp += " Expected -- head not tracking proximity."

    per_seed_rows = "\n".join(
        f"  seed={r['seed']}:"
        f" probe_r2={r['probe_r2']:.4f}"
        f" head_r={r['benefit_head_r']:.4f}"
        f" head_r_p1={r['head_r_p1']:.4f}"
        f" on={r['benefit_per_ep_on']:.3f}"
        f" off={r['benefit_per_ep_off']:.3f}"
        f" ratio={r['benefit_ratio']:.2f}x"
        f" loss={r['final_benefit_loss']:.6f}"
        f" C1={'PASS' if r['c1_pass'] else 'FAIL'}"
        f" C2={'PASS' if r['c2_pass'] else 'FAIL'}"
        f" C3={'PASS' if r['c3_pass'] else 'FAIL'}"
        for r in all_results
    )

    print(f"\n{'='*60}", flush=True)
    print(f"[V3-EXQ-180a] FINAL: {status}", flush=True)
    print(f"  {c1_interp}", flush=True)
    print(f"  {c2_interp}", flush=True)
    print(f"  {c3_interp}", flush=True)
    print(f"\nPer-seed:\n{per_seed_rows}", flush=True)
    print(f"{'='*60}", flush=True)

    summary_markdown = (
        f"# V3-EXQ-180a -- Resource Proximity benefit_eval_head with Phased Training\n\n"
        f"**Status:** {status}\n"
        f"**Claims:** ARC-030\n"
        f"**Seeds:** {list(seeds)}\n"
        f"**Supersedes:** V3-EXQ-180\n\n"
        f"## Context\n\n"
        f"EXQ-180 showed probe_r2=0.619 (z_world DOES encode resource proximity, C1 PASS)"
        f" but benefit_ratio=0.29x (C3 FAIL). Root cause: joint training -- benefit_eval_head"
        f" chased moving z_world during E1 convergence. Fix: phased training.\n\n"
        f"## Design (Phased Training Protocol)\n\n"
        f"**Phase 0 ({p0_episodes} eps):** E1+E2 warmup ONLY. No benefit_eval training."
        f" Let z_world converge.\n\n"
        f"**Phase 1 ({p1_episodes} eps):** Freeze E1+E2. Train benefit_eval_head on"
        f" z_world.detach() with MSE on resource_field_view.max() (continuous proximity)."
        f" Also fit linear probe for R^2 diagnostic.\n\n"
        f"**Phase 2 ({p2_episodes} eps):** Eval with benefit_eval_enabled=True in E3"
        f" trajectory scoring. BENEFIT_EVAL_ON vs BENEFIT_EVAL_OFF (random).\n\n"
        f"**Config:** alpha_world={alpha_world}, benefit_weight={benefit_weight},"
        f" num_hazards={num_hazards}, num_resources={num_resources},"
        f" use_proxy_fields=True\n\n"
        f"## Key Results\n\n"
        f"| Metric | Value | Threshold |\n"
        f"|---|---|---|\n"
        f"| probe_r2 (linear) | {avg_probe_r2:.4f} | > 0.3 |\n"
        f"| benefit_head_r (eval) | {avg_head_r:.4f} | > 0.3 |\n"
        f"| head_r_p1 (training) | {avg_head_r_p1:.4f} | -- (diagnostic) |\n"
        f"| benefit_ratio | {avg_ratio:.2f}x | > 1.0 |\n"
        f"| benefit_on | {avg_benefit_on:.3f} | -- |\n"
        f"| benefit_off | {avg_benefit_off:.3f} | -- |\n"
        f"| final_benefit_loss | {avg_loss:.6f} | -- |\n\n"
        f"## Pass Criteria\n\n"
        f"| Criterion | Result | Seeds passing |\n"
        f"|---|---|---|\n"
        f"| C1: probe_r2 > 0.3 | {'PASS' if c1_overall else 'FAIL'} | {seeds_c1}/{n_seeds} |\n"
        f"| C2: benefit_head_r > 0.3 | {'PASS' if c2_overall else 'FAIL'} | {seeds_c2}/{n_seeds} |\n"
        f"| C3: benefit_ratio > 1.0 | {'PASS' if c3_overall else 'FAIL'} | {seeds_c3}/{n_seeds} |\n\n"
        f"PASS rule: all three criteria pass in >= {pass_threshold}/{n_seeds} seeds"
        f" -> **{status}**\n\n"
        f"## Interpretation\n\n"
        f"{c1_interp}\n\n{c2_interp}\n\n{c3_interp}\n\n"
        f"## Per-Seed Results\n\n{per_seed_rows}\n"
    )

    evidence_direction = (
        "supports" if all_pass
        else ("mixed" if (seeds_c1 + seeds_c2 + seeds_c3) >= n_seeds else "weakens")
    )

    metrics = {
        "probe_r2": float(avg_probe_r2),
        "benefit_head_r": float(avg_head_r),
        "head_r_p1": float(avg_head_r_p1),
        "benefit_per_ep_on": float(avg_benefit_on),
        "benefit_per_ep_off": float(avg_benefit_off),
        "benefit_ratio": float(avg_ratio),
        "final_benefit_loss": float(avg_loss),
        "crit1_pass": 1.0 if c1_overall else 0.0,
        "crit2_pass": 1.0 if c2_overall else 0.0,
        "crit3_pass": 1.0 if c3_overall else 0.0,
        "seeds_c1_pass": float(seeds_c1),
        "seeds_c2_pass": float(seeds_c2),
        "seeds_c3_pass": float(seeds_c3),
        "n_seeds": float(n_seeds),
        "alpha_world": float(alpha_world),
        "benefit_weight": float(benefit_weight),
        "num_hazards": float(num_hazards),
        "num_resources": float(num_resources),
        "p0_episodes": float(p0_episodes),
        "p1_episodes": float(p1_episodes),
        "p2_episodes": float(p2_episodes),
    }

    return {
        "status": status,
        "metrics": metrics,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": evidence_direction,
        "experiment_type": EXPERIMENT_TYPE,
        "fatal_error_count": 0,
        "supersedes": "v3_exq_180_resource_prox_gradient_diag",
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 11])
    parser.add_argument("--p0-eps", type=int, default=200)
    parser.add_argument("--p1-eps", type=int, default=100)
    parser.add_argument("--p2-eps", type=int, default=200)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--alpha-world", type=float, default=0.9)
    parser.add_argument("--alpha-self", type=float, default=0.3)
    parser.add_argument("--harm-scale", type=float, default=0.02)
    parser.add_argument("--proximity-scale", type=float, default=0.05)
    parser.add_argument("--novelty-weight", type=float, default=0.1)
    parser.add_argument("--benefit-weight", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--num-hazards", type=int, default=4)
    parser.add_argument("--num-resources", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true",
                        help="Run 1 seed, tiny episodes for smoke test")
    args = parser.parse_args()

    if args.dry_run:
        args.seeds = [42]
        args.p0_eps = 5
        args.p1_eps = 3
        args.p2_eps = 5
        args.steps = 50
        print("[DRY-RUN] 1 seed, P0=5 P1=3 P2=5 x 50 steps", flush=True)

    result = run(
        seeds=tuple(args.seeds),
        p0_episodes=args.p0_eps,
        p1_episodes=args.p1_eps,
        p2_episodes=args.p2_eps,
        steps_per_episode=args.steps,
        alpha_world=args.alpha_world,
        alpha_self=args.alpha_self,
        harm_scale=args.harm_scale,
        proximity_harm_scale=args.proximity_scale,
        novelty_bonus_weight=args.novelty_weight,
        benefit_weight=args.benefit_weight,
        temperature=args.temperature,
        num_hazards=args.num_hazards,
        num_resources=args.num_resources,
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
