"""
V3-EXQ-104b -- E1/E2 Parallel Rollout vs Frozen z_world (trained agent) [MECH-135 diagnostic]

Claims: MECH-135

Supersedes: V3-EXQ-104 (untrained agent -- implementation artifact)

Root cause of EXQ-104 FAIL: agent had random weights (no training phase). With random weights:
  - E2.world_forward maps inputs toward near-zero immediately (untrained MLP output).
  - E2_WORLD goal scores collapse to ~4 (near-zero world states have low proximity to goal).
  - FROZEN scores ~30 = 30 steps x ~1/step from high-dim random cosine similarity (random
    vectors in 32-dim space have cosine sim ~0.18-0.25 -- uniformly positive, not zero).
  - This creates a spurious FROZEN > E2_WORLD gap with no scientific content.

Fix: add Phase 0 training (N episodes) before rollout test. After training:
  - E2.world_forward produces meaningful z_world transitions.
  - E1 predictions have learned z_world structure.
  - Goal template from resource contact represents a genuine attractor region.
  - FROZEN vs E2_WORLD vs E1_PRED comparison is now scientifically interpretable.

MECH-135 predicts: after training, E1_PRED or E2_WORLD should produce HIGHER goal scores
than FROZEN because parallel z_world evolution allows E3 to detect goal-proximity changes
during rollout. Frozen z_world cannot detect resource contact in the simulated trajectory.

Design (same 3-condition structure as EXQ-104, adds Phase 0 training):
  Phase 0: Train agent for n_train_episodes episodes.
  Phase 1: Collect goal template (resource contact z_world).
  Phase 2: Generate 60 rollouts per condition (FROZEN / E2_WORLD / E1_PRED), horizon=30.
  Phase 3: Score trajectories, apply pass criteria.

Pass criteria (same as EXQ-104):
  C1: goal_score_variance(best_parallel) > 10 * goal_score_variance(FROZEN)
  C2: mean goal_score(best_parallel) > 1.05 * mean goal_score(FROZEN)
  C3: prox_delta_pos_frac > 0.40 for best_parallel condition

Diagnostic metric (new in 104b):
  D1: log mean z_world norm in each condition at each rollout step.
      If E2_WORLD z_world norms collapse toward 0 even after training, E2.world_forward
      has a distribution-shift problem. If norms stay stable, the collapse in EXQ-104
      was purely a training artifact.
"""

import sys
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.goal import GoalConfig, GoalState
from ree_core.predictors.e2_fast import Trajectory
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_104b_e1_parallel_rollout_trained"
CLAIM_IDS = ["MECH-135"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _build_agent(seed: int, world_dim: int, self_dim: int) -> Tuple[REEAgent, CausalGridWorldV2]:
    env = CausalGridWorldV2(
        seed=seed, size=10, num_hazards=2, num_resources=4,
        hazard_harm=0.02, env_drift_interval=8, env_drift_prob=0.05,
        proximity_harm_scale=0.03, proximity_benefit_scale=0.04,
        proximity_approach_threshold=0.15, hazard_field_decay=0.5,
        resource_respawn_on_consume=True,
    )
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=0.9,
        alpha_self=0.3,
    )
    config.latent.unified_latent_mode = False
    agent = REEAgent(config)
    return agent, env


# ---------------------------------------------------------------------------
# Phase 0: train agent for N episodes
# ---------------------------------------------------------------------------

def _train_agent(
    agent: REEAgent,
    env: CausalGridWorldV2,
    seed: int,
    n_episodes: int,
    steps_per_episode: int,
) -> None:
    """Train agent for n_episodes using random policy. Creates local optimizers."""
    torch.manual_seed(seed + 2000)
    random.seed(seed + 2000)
    agent.train()

    opt_e1 = optim.Adam(agent.e1.parameters(), lr=1e-3)
    opt_e2 = optim.Adam(agent.e2.parameters(), lr=1e-3)

    for ep in range(n_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        ep_loss_e1 = 0.0
        ep_loss_e2 = 0.0
        n_steps = 0

        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]

            with torch.no_grad():
                latent_prev = agent.sense(obs_body, obs_world)
            agent.clock.advance()

            action_idx = random.randint(0, env.action_dim - 1)
            action = _action_to_onehot(action_idx, env.action_dim, agent.device)
            _, _, done, _, obs_dict = env.step(action)

            obs_body_next = obs_dict["body_state"]
            obs_world_next = obs_dict["world_state"]
            with torch.no_grad():
                latent_next = agent.sense(obs_body_next, obs_world_next)

            # E1 update (world prediction error)
            opt_e1.zero_grad()
            total_prev = torch.cat([latent_prev.z_self, latent_prev.z_world], dim=-1)
            total_next = torch.cat([latent_next.z_self, latent_next.z_world], dim=-1)
            e1_pred, _ = agent.e1(total_prev, horizon=1)
            e1_loss = F.mse_loss(e1_pred[:, 0, :], total_next.detach())
            e1_loss.backward()
            opt_e1.step()
            ep_loss_e1 += e1_loss.item()

            # E2 update (motor-sensory prediction error on z_self)
            opt_e2.zero_grad()
            z_self_pred = agent.e2.predict_next_self(latent_prev.z_self.detach(), action)
            e2_loss = F.mse_loss(z_self_pred, latent_next.z_self.detach())
            e2_loss.backward()
            opt_e2.step()
            ep_loss_e2 += e2_loss.item()

            n_steps += 1
            if done:
                break

        if (ep + 1) % 20 == 0:
            print(
                f"  [Train] ep {ep+1}/{n_episodes} "
                f"e1_loss={ep_loss_e1/max(n_steps,1):.5f} "
                f"e2_loss={ep_loss_e2/max(n_steps,1):.5f}",
                flush=True,
            )

    agent.eval()
    print(f"  [Train] Done. {n_episodes} episodes.", flush=True)


# ---------------------------------------------------------------------------
# Phase 1: collect resource-contact z_world as goal template
# ---------------------------------------------------------------------------

def _collect_goal_template(
    agent: REEAgent,
    env: CausalGridWorldV2,
    seed: int,
    max_steps: int,
) -> Optional[torch.Tensor]:
    torch.manual_seed(seed)
    random.seed(seed)
    _, obs_dict = env.reset()
    agent.reset()

    for _ in range(max_steps):
        obs_body = obs_dict["body_state"]
        obs_world = obs_dict["world_state"]
        with torch.no_grad():
            latent = agent.sense(obs_body, obs_world)
        agent.clock.advance()
        action_idx = random.randint(0, env.action_dim - 1)
        action = _action_to_onehot(action_idx, env.action_dim, agent.device)
        _, _, done, info, obs_dict = env.step(action)
        if info.get("transition_type", "none") == "resource":
            print(
                f"  [Phase1] Resource contact, z_world_norm={latent.z_world.norm().item():.3f}",
                flush=True,
            )
            return latent.z_world.detach()
        if done:
            _, obs_dict = env.reset()
            agent.reset()

    print("  [Phase1] WARNING: no resource contact -- using random unit vector", flush=True)
    z_goal = torch.randn(1, agent.config.latent.world_dim)
    z_goal = F.normalize(z_goal, dim=-1)
    return z_goal


# ---------------------------------------------------------------------------
# Phase 2: generate rollouts
# ---------------------------------------------------------------------------

def _get_start_state(
    agent: REEAgent,
    env: CausalGridWorldV2,
    seed: int,
    n_warmup_steps: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed + 1000)
    random.seed(seed + 1000)
    _, obs_dict = env.reset()
    agent.reset()
    latent = None
    for _ in range(n_warmup_steps):
        obs_body = obs_dict["body_state"]
        obs_world = obs_dict["world_state"]
        with torch.no_grad():
            latent = agent.sense(obs_body, obs_world)
        agent.clock.advance()
        action_idx = random.randint(0, env.action_dim - 1)
        action = _action_to_onehot(action_idx, env.action_dim, agent.device)
        _, _, done, _, obs_dict = env.step(action)
        if done:
            _, obs_dict = env.reset()
            agent.reset()
            latent = None
    if latent is None:
        obs_body = obs_dict["body_state"]
        obs_world = obs_dict["world_state"]
        with torch.no_grad():
            latent = agent.sense(obs_body, obs_world)
    z_self = latent.z_self.detach()
    z_world = latent.z_world.detach()
    total = torch.cat([z_self, z_world], dim=-1)
    return z_self, z_world, total


def _generate_rollouts(
    agent: REEAgent,
    env: CausalGridWorldV2,
    seed: int,
    rollout_horizon: int,
    n_rollouts: int,
    n_warmup_steps: int,
    condition: str,
) -> Tuple[List[Trajectory], List[float]]:
    """
    Generate rollouts under condition: 'FROZEN', 'E2_WORLD', or 'E1_PRED'.
    Returns trajectories and per-step mean z_world norms (diagnostic D1).
    """
    device = agent.device
    self_dim = agent.config.latent.self_dim
    world_dim = agent.config.latent.world_dim
    trajectories = []
    world_norm_sums = [0.0] * (rollout_horizon + 1)
    world_norm_counts = [0] * (rollout_horizon + 1)

    for r in range(n_rollouts):
        z_self, z_world, total = _get_start_state(agent, env, seed + r * 7, n_warmup_steps)

        z_self_states = [z_self]
        z_world_states = [z_world]
        actions_list = []

        z_self_curr = z_self
        z_world_curr = z_world

        world_norm_sums[0] += z_world.norm().item()
        world_norm_counts[0] += 1

        if condition == "E1_PRED":
            with torch.no_grad():
                e1_preds, _ = agent.e1(total, horizon=rollout_horizon)

        for step in range(rollout_horizon):
            a_idx = random.randint(0, env.action_dim - 1)
            action = _action_to_onehot(a_idx, env.action_dim, device)
            with torch.no_grad():
                z_self_next = agent.e2.predict_next_self(z_self_curr, action)

            if condition == "FROZEN":
                z_world_next = z_world  # frozen at t=0
            elif condition == "E2_WORLD":
                with torch.no_grad():
                    z_world_next = agent.e2.world_forward(z_world_curr, action)
            else:  # E1_PRED
                e1_step = e1_preds[0, step, :]
                z_world_next = e1_step[self_dim:].unsqueeze(0)

            z_self_curr = z_self_next
            z_world_curr = z_world_next
            z_self_states.append(z_self_curr)
            z_world_states.append(z_world_curr)
            actions_list.append(action)

            world_norm_sums[step + 1] += z_world_curr.norm().item()
            world_norm_counts[step + 1] += 1

        actions_tensor = torch.cat(actions_list, dim=0).unsqueeze(0)
        traj = Trajectory(
            states=z_self_states,
            actions=actions_tensor,
            world_states=z_world_states,
        )
        trajectories.append(traj)

    norm_profile = [
        world_norm_sums[i] / max(world_norm_counts[i], 1)
        for i in range(rollout_horizon + 1)
    ]
    return trajectories, norm_profile


# ---------------------------------------------------------------------------
# Phase 3: score trajectories
# ---------------------------------------------------------------------------

def _score_trajectories(
    trajectories: List[Trajectory],
    goal_state: GoalState,
    e3,
) -> Dict[str, float]:
    scores = []
    deltas = []
    for traj in trajectories:
        with torch.no_grad():
            score = e3.compute_goal_score(traj, goal_state)
        scores.append(float(score.mean().item()))
        if traj.world_states is not None and len(traj.world_states) >= 2:
            with torch.no_grad():
                prox_init = goal_state.goal_proximity(traj.world_states[0])
                prox_final = goal_state.goal_proximity(traj.world_states[-1])
            delta = float((prox_final - prox_init).mean().item())
            deltas.append(delta)
    scores_t = torch.tensor(scores)
    pos_frac = float(sum(d > 0 for d in deltas) / max(len(deltas), 1))
    return {
        "goal_score_mean": float(scores_t.mean().item()),
        "goal_score_var": float(scores_t.var().item()) if len(scores) > 1 else 0.0,
        "prox_delta_pos_frac": pos_frac,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    seed: int,
    world_dim: int,
    self_dim: int,
    n_train_episodes: int,
    steps_per_episode: int,
    rollout_horizon: int,
    n_rollouts: int,
    n_warmup_steps: int,
    goal_collect_max_steps: int,
    c1_variance_ratio: float,
    c2_score_ratio: float,
    c3_pos_frac_threshold: float,
) -> Dict:
    torch.manual_seed(seed)
    random.seed(seed)

    print(
        f"[EXQ-104b] train_eps={n_train_episodes}, steps/ep={steps_per_episode}, "
        f"rollout_horizon={rollout_horizon}, n_rollouts={n_rollouts}",
        flush=True,
    )

    agent, env = _build_agent(seed, world_dim, self_dim)

    # Phase 0: train
    print("[EXQ-104b] Phase 0: training agent...", flush=True)
    _train_agent(agent, env, seed, n_train_episodes, steps_per_episode)

    # Phase 1: goal template
    print("[EXQ-104b] Phase 1: collecting goal template...", flush=True)
    z_goal_template = _collect_goal_template(agent, env, seed, goal_collect_max_steps)
    goal_config = GoalConfig(goal_dim=world_dim, z_goal_enabled=True, goal_weight=1.0)
    goal_state = GoalState(goal_config, agent.device)
    goal_state._z_goal = z_goal_template.to(agent.device)
    print(f"  z_goal norm={goal_state.goal_norm():.4f}", flush=True)

    # Phase 2: rollouts
    print("[EXQ-104b] Phase 2: generating rollouts...", flush=True)
    trajs_frozen, norms_frozen = _generate_rollouts(
        agent, env, seed, rollout_horizon, n_rollouts, n_warmup_steps, "FROZEN"
    )
    trajs_e2, norms_e2 = _generate_rollouts(
        agent, env, seed, rollout_horizon, n_rollouts, n_warmup_steps, "E2_WORLD"
    )
    trajs_e1, norms_e1 = _generate_rollouts(
        agent, env, seed, rollout_horizon, n_rollouts, n_warmup_steps, "E1_PRED"
    )

    # Phase 3: score
    e3 = agent.e3
    m_frozen = _score_trajectories(trajs_frozen, goal_state, e3)
    m_e2 = _score_trajectories(trajs_e2, goal_state, e3)
    m_e1 = _score_trajectories(trajs_e1, goal_state, e3)

    print(f"  FROZEN:   mean={m_frozen['goal_score_mean']:.4f} var={m_frozen['goal_score_var']:.6f} pos={m_frozen['prox_delta_pos_frac']:.3f}", flush=True)
    print(f"  E2_WORLD: mean={m_e2['goal_score_mean']:.4f} var={m_e2['goal_score_var']:.6f} pos={m_e2['prox_delta_pos_frac']:.3f}", flush=True)
    print(f"  E1_PRED:  mean={m_e1['goal_score_mean']:.4f} var={m_e1['goal_score_var']:.6f} pos={m_e1['prox_delta_pos_frac']:.3f}", flush=True)

    # D1: z_world norm profile (diagnostic)
    print(f"  D1 z_world norm profiles (step 0/5/10/15/20/25/30):", flush=True)
    steps_to_log = [0, 5, 10, 15, 20, 25, 30]
    for s in steps_to_log:
        if s < len(norms_frozen):
            print(
                f"    step {s:2d}: FROZEN={norms_frozen[s]:.3f} E2={norms_e2[s]:.3f} E1={norms_e1[s]:.3f}",
                flush=True,
            )

    best_parallel_var = max(m_e2["goal_score_var"], m_e1["goal_score_var"])
    best_parallel_score = max(m_e2["goal_score_mean"], m_e1["goal_score_mean"])
    best_parallel_pos = max(m_e2["prox_delta_pos_frac"], m_e1["prox_delta_pos_frac"])
    frozen_var = max(m_frozen["goal_score_var"], 1e-9)

    var_ratio = best_parallel_var / frozen_var
    score_ratio = best_parallel_score / max(m_frozen["goal_score_mean"], 1e-9)

    c1 = var_ratio >= c1_variance_ratio
    c2 = score_ratio >= c2_score_ratio
    c3 = best_parallel_pos >= c3_pos_frac_threshold

    print(f"\n[EXQ-104b] C1 (var_ratio>={c1_variance_ratio}): {c1} ratio={var_ratio:.2f}", flush=True)
    print(f"[EXQ-104b] C2 (score_ratio>={c2_score_ratio}): {c2} ratio={score_ratio:.3f}", flush=True)
    print(f"[EXQ-104b] C3 (pos_frac>={c3_pos_frac_threshold}): {c3} frac={best_parallel_pos:.3f}", flush=True)

    status = "PASS" if (c1 and c2 and c3) else "FAIL"
    print(f"[EXQ-104b] Status: {status}", flush=True)

    return {
        "experiment_type": EXPERIMENT_TYPE,
        "claim_ids": CLAIM_IDS,
        "seed": seed,
        "n_train_episodes": n_train_episodes,
        "steps_per_episode": steps_per_episode,
        "rollout_horizon": rollout_horizon,
        "n_rollouts": n_rollouts,
        "world_dim": world_dim,
        "self_dim": self_dim,
        "z_goal_norm": goal_state.goal_norm(),
        "frozen_goal_score_mean": m_frozen["goal_score_mean"],
        "frozen_goal_score_var": m_frozen["goal_score_var"],
        "frozen_prox_delta_pos": m_frozen["prox_delta_pos_frac"],
        "e2world_goal_score_mean": m_e2["goal_score_mean"],
        "e2world_goal_score_var": m_e2["goal_score_var"],
        "e2world_prox_delta_pos": m_e2["prox_delta_pos_frac"],
        "e1pred_goal_score_mean": m_e1["goal_score_mean"],
        "e1pred_goal_score_var": m_e1["goal_score_var"],
        "e1pred_prox_delta_pos": m_e1["prox_delta_pos_frac"],
        "variance_ratio": var_ratio,
        "score_ratio": score_ratio,
        "best_parallel_pos_frac": best_parallel_pos,
        "d1_norm_frozen_step0": norms_frozen[0] if norms_frozen else 0.0,
        "d1_norm_frozen_step10": norms_frozen[10] if len(norms_frozen) > 10 else 0.0,
        "d1_norm_frozen_step30": norms_frozen[-1] if norms_frozen else 0.0,
        "d1_norm_e2_step0": norms_e2[0] if norms_e2 else 0.0,
        "d1_norm_e2_step10": norms_e2[10] if len(norms_e2) > 10 else 0.0,
        "d1_norm_e2_step30": norms_e2[-1] if norms_e2 else 0.0,
        "d1_norm_e1_step0": norms_e1[0] if norms_e1 else 0.0,
        "d1_norm_e1_step10": norms_e1[10] if len(norms_e1) > 10 else 0.0,
        "d1_norm_e1_step30": norms_e1[-1] if norms_e1 else 0.0,
        "c1_variance_ratio_pass": bool(c1),
        "c2_score_ratio_pass": bool(c2),
        "c3_pos_frac_pass": bool(c3),
        "criteria_met": sum([c1, c2, c3]),
        "criteria_total": 3,
        "status": status,
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser(
        description="V3-EXQ-104b: parallel rollout with trained agent (MECH-135 diagnostic)"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--world-dim", type=int, default=32)
    parser.add_argument("--self-dim", type=int, default=32)
    parser.add_argument("--train-episodes", type=int, default=100)
    parser.add_argument("--steps-per-episode", type=int, default=200)
    parser.add_argument("--rollout-horizon", type=int, default=30)
    parser.add_argument("--n-rollouts", type=int, default=60)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--goal-max-steps", type=int, default=2000)
    parser.add_argument("--c1-var-ratio", type=float, default=10.0)
    parser.add_argument("--c2-score-ratio", type=float, default=1.05)
    parser.add_argument("--c3-pos-frac", type=float, default=0.40)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        n_train = 5
        steps_ep = 50
        n_rollouts = 5
        warmup = 10
        goal_max = 300
        c1, c2, c3 = 1.0, 1.0, 0.0
        print("[V3-EXQ-104b] SMOKE TEST MODE", flush=True)
    else:
        n_train = args.train_episodes
        steps_ep = args.steps_per_episode
        n_rollouts = args.n_rollouts
        warmup = args.warmup_steps
        goal_max = args.goal_max_steps
        c1, c2, c3 = args.c1_var_ratio, args.c2_score_ratio, args.c3_pos_frac

    result = run(
        seed=args.seed,
        world_dim=args.world_dim,
        self_dim=args.self_dim,
        n_train_episodes=n_train,
        steps_per_episode=steps_ep,
        rollout_horizon=args.rollout_horizon,
        n_rollouts=n_rollouts,
        n_warmup_steps=warmup,
        goal_collect_max_steps=goal_max,
        c1_variance_ratio=c1,
        c2_score_ratio=c2,
        c3_pos_frac_threshold=c3,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"] = ts
    result["run_id"] = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = "ree_hybrid_guardrails_v1"
    result["verdict"] = result["status"]

    out_dir = (
        Path(__file__).resolve().parents[2]
        / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{EXPERIMENT_TYPE}_{ts}.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"\nResult written to: {out_path}", flush=True)
    print(f"Status: {result['status']}", flush=True)

    if args.smoke_test:
        print("[V3-EXQ-104b] SMOKE TEST COMPLETE", flush=True)
        for k in ["variance_ratio", "score_ratio", "best_parallel_pos_frac",
                  "d1_norm_e2_step0", "d1_norm_e2_step30", "criteria_met"]:
            print(f"  {k}: {result.get(k, 'N/A')}", flush=True)
