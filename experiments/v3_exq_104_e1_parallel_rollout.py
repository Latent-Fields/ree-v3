"""
V3-EXQ-104 -- E1/E2 Parallel Rollout vs Frozen z_world (MECH-135 core test)

Claims: MECH-135

MECH-135 asserts: if z_world is frozen during E2's planning rollout, E3 evaluates goal
achievement against a stale world state, suppressing goal-directed trajectory selection.
The fix: E1 (cortical) or E2.world_forward co-evolves z_world alongside E2's z_self
update at each rollout step.

Neurological basis: cerebro-cerebellar parallel forward simulation -- cerebellar (E2)
and cortical (E1) forward models co-evolve during imagined movement.

Design:
  Forced goal seeding: directly set GoalState._z_goal from a resource-contact z_world
  vector (collected from a short warmup run). This bypasses the SD-012 drive/wiring issue
  and isolates the rollout question.

  Three conditions:
    A (FROZEN):   z_world held constant at t=0 throughout 30-step rollout.
    B (E2_WORLD): z_world updated each step via E2.world_forward(z_world_t, a_t).
    C (E1_PRED):  z_world updated each step from E1's action-unconditional predictions
                  (z_world component of E1.forward(total_state)[:,t,self_dim:]).

  For each condition: 60 rollouts of 30 steps each, from random start states.
  Trajectories stored as Trajectory objects with world_states populated appropriately.
  E3.compute_goal_score() computed for each trajectory.

Pass criteria (ALL required):
  C1 (primary): goal_score_variance in condition B or C > 10 * goal_score_variance in A.
     Frozen z_world produces near-zero variance in goal scores across rollout steps.
     Parallel rollout produces variance > 0 (E3 can distinguish trajectories by goal proximity).
  C2: mean goal_score(B or C) > 1.05 * mean goal_score(A).
     Parallel rollout finds trajectories that look more goal-relevant on average.
  C3: goal_proximity_delta > 0 for at least 40% of goal-directed rollouts in B or C.
     goal_proximity_delta = goal_proximity(final_state) - goal_proximity(initial_state).
     Positive = trajectory approaches goal in z_world space.

FAIL means: frozen z_world is equally good at detecting goal proximity as parallel rollout.
This would weaken MECH-135 (parallel rollout may not matter for goal visibility).
"""

import sys
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.goal import GoalConfig, GoalState
from ree_core.predictors.e2_fast import Trajectory
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_104_e1_parallel_rollout"
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
# Phase 1: collect resource-contact z_world as goal template
# ---------------------------------------------------------------------------

def _collect_goal_template(
    agent: REEAgent,
    env: CausalGridWorldV2,
    seed: int,
    max_steps: int,
) -> Optional[torch.Tensor]:
    """
    Run agent until it contacts a resource. Record z_world at that moment.
    Returns z_world [1, world_dim] or None if no contact in max_steps.
    """
    torch.manual_seed(seed)
    random.seed(seed)

    _, obs_dict = env.reset()
    agent.reset()

    for _ in range(max_steps):
        obs_body  = obs_dict["body_state"]
        obs_world = obs_dict["world_state"]

        with torch.no_grad():
            latent = agent.sense(obs_body, obs_world)
        agent.clock.advance()

        action_idx = random.randint(0, env.action_dim - 1)
        action = _action_to_onehot(action_idx, env.action_dim, agent.device)

        _, _, done, info, obs_dict = env.step(action)
        ttype = info.get("transition_type", "none")

        if ttype == "resource":
            print(
                f"  [Phase1] Resource contact at step, z_world_norm="
                f"{latent.z_world.norm().item():.3f}",
                flush=True,
            )
            return latent.z_world.detach()

        if done:
            _, obs_dict = env.reset()
            agent.reset()

    print("  [Phase1] WARNING: no resource contact found in max_steps", flush=True)
    # Fall back: return random unit vector as goal template
    z_goal = torch.randn(1, agent.config.latent.world_dim)
    z_goal = F.normalize(z_goal, dim=-1)
    return z_goal


# ---------------------------------------------------------------------------
# Phase 2: generate rollouts under each condition
# ---------------------------------------------------------------------------

def _get_start_state(
    agent: REEAgent,
    env: CausalGridWorldV2,
    seed: int,
    n_warmup_steps: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run N warmup steps, return (z_self, z_world, total_state) at end.
    """
    torch.manual_seed(seed + 1000)
    random.seed(seed + 1000)

    _, obs_dict = env.reset()
    agent.reset()
    latent = None

    for _ in range(n_warmup_steps):
        obs_body  = obs_dict["body_state"]
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
        # safety: get one final sense
        obs_body  = obs_dict["body_state"]
        obs_world = obs_dict["world_state"]
        with torch.no_grad():
            latent = agent.sense(obs_body, obs_world)

    z_self  = latent.z_self.detach()
    z_world = latent.z_world.detach()
    total   = torch.cat([z_self, z_world], dim=-1)
    return z_self, z_world, total


def _generate_rollouts_frozen(
    agent: REEAgent,
    env: CausalGridWorldV2,
    seed: int,
    rollout_horizon: int,
    n_rollouts: int,
    n_warmup_steps: int,
) -> List[Trajectory]:
    """Condition A: z_world frozen throughout rollout."""
    device = agent.device
    trajectories = []

    for r in range(n_rollouts):
        z_self, z_world, _ = _get_start_state(agent, env, seed + r * 7, n_warmup_steps)

        z_self_states  = [z_self]
        z_world_states = [z_world]   # frozen: same z_world repeated
        actions_list   = []

        z_curr = z_self
        for _ in range(rollout_horizon):
            a_idx  = random.randint(0, env.action_dim - 1)
            action = _action_to_onehot(a_idx, env.action_dim, device)
            with torch.no_grad():
                z_next = agent.e2.predict_next_self(z_curr, action)
            z_curr = z_next
            z_self_states.append(z_curr)
            z_world_states.append(z_world)   # frozen: reuse t=0 world state
            actions_list.append(action)

        actions_tensor = torch.cat(actions_list, dim=0).unsqueeze(0)  # [1, horizon, action_dim]
        traj = Trajectory(
            states       = z_self_states,
            actions      = actions_tensor,
            world_states = z_world_states,
        )
        trajectories.append(traj)

    return trajectories


def _generate_rollouts_e2_world(
    agent: REEAgent,
    env: CausalGridWorldV2,
    seed: int,
    rollout_horizon: int,
    n_rollouts: int,
    n_warmup_steps: int,
) -> List[Trajectory]:
    """Condition B: z_world updated via E2.world_forward at each rollout step."""
    device = agent.device
    trajectories = []

    for r in range(n_rollouts):
        z_self, z_world, _ = _get_start_state(agent, env, seed + r * 7, n_warmup_steps)

        z_self_states  = [z_self]
        z_world_states = [z_world]
        actions_list   = []

        z_self_curr  = z_self
        z_world_curr = z_world

        for _ in range(rollout_horizon):
            a_idx  = random.randint(0, env.action_dim - 1)
            action = _action_to_onehot(a_idx, env.action_dim, device)
            with torch.no_grad():
                z_self_next  = agent.e2.predict_next_self(z_self_curr, action)
                z_world_next = agent.e2.world_forward(z_world_curr, action)
            z_self_curr  = z_self_next
            z_world_curr = z_world_next
            z_self_states.append(z_self_curr)
            z_world_states.append(z_world_curr)
            actions_list.append(action)

        actions_tensor = torch.cat(actions_list, dim=0).unsqueeze(0)
        traj = Trajectory(
            states       = z_self_states,
            actions      = actions_tensor,
            world_states = z_world_states,
        )
        trajectories.append(traj)

    return trajectories


def _generate_rollouts_e1_pred(
    agent: REEAgent,
    env: CausalGridWorldV2,
    seed: int,
    rollout_horizon: int,
    n_rollouts: int,
    n_warmup_steps: int,
) -> List[Trajectory]:
    """Condition C: z_world updated from E1's action-unconditional step predictions."""
    device    = agent.device
    self_dim  = agent.config.latent.self_dim
    world_dim = agent.config.latent.world_dim
    trajectories = []

    for r in range(n_rollouts):
        z_self, z_world, total_state = _get_start_state(agent, env, seed + r * 7, n_warmup_steps)

        # E1: predict horizon steps ahead, action-unconditional
        with torch.no_grad():
            e1_preds, _ = agent.e1(total_state, horizon=rollout_horizon)
        # e1_preds: [1, rollout_horizon, self_dim + world_dim]

        z_self_states  = [z_self]
        z_world_states = [z_world]
        actions_list   = []

        z_self_curr = z_self
        for step in range(rollout_horizon):
            a_idx  = random.randint(0, env.action_dim - 1)
            action = _action_to_onehot(a_idx, env.action_dim, device)
            with torch.no_grad():
                z_self_next = agent.e2.predict_next_self(z_self_curr, action)
            z_self_curr = z_self_next

            # z_world from E1 prediction at this horizon step
            e1_step = e1_preds[0, step, :]         # [self_dim + world_dim]
            z_world_e1 = e1_step[self_dim:].unsqueeze(0)   # [1, world_dim]

            z_self_states.append(z_self_curr)
            z_world_states.append(z_world_e1)
            actions_list.append(action)

        actions_tensor = torch.cat(actions_list, dim=0).unsqueeze(0)
        traj = Trajectory(
            states       = z_self_states,
            actions      = actions_tensor,
            world_states = z_world_states,
        )
        trajectories.append(traj)

    return trajectories


# ---------------------------------------------------------------------------
# Phase 3: score trajectories and compute metrics
# ---------------------------------------------------------------------------

def _score_trajectories(
    trajectories: List[Trajectory],
    goal_state: GoalState,
    e3,
) -> Dict[str, float]:
    """
    Compute goal-related metrics across a set of trajectories.

    Returns:
        goal_score_mean: mean E3 goal score
        goal_score_var:  variance in E3 goal scores
        prox_delta_pos_frac: fraction of trajectories where final step
                             goal_proximity > initial step goal_proximity
    """
    scores    = []
    deltas    = []

    for traj in trajectories:
        # E3 goal score (cumulative goal proximity across rollout)
        with torch.no_grad():
            score = e3.compute_goal_score(traj, goal_state)
        scores.append(float(score.mean().item()))

        # Goal proximity at first vs last world state
        if traj.world_states is not None and len(traj.world_states) >= 2:
            with torch.no_grad():
                prox_init  = goal_state.goal_proximity(traj.world_states[0])
                prox_final = goal_state.goal_proximity(traj.world_states[-1])
            delta = float((prox_final - prox_init).mean().item())
            deltas.append(delta)

    scores_t = torch.tensor(scores)
    pos_frac = float(sum(d > 0 for d in deltas) / max(len(deltas), 1))

    return {
        "goal_score_mean": float(scores_t.mean().item()),
        "goal_score_var":  float(scores_t.var().item()) if len(scores) > 1 else 0.0,
        "prox_delta_pos_frac": pos_frac,
        "n_trajectories": len(trajectories),
    }


# ---------------------------------------------------------------------------
# Main run function
# ---------------------------------------------------------------------------

def run(
    seed: int,
    world_dim: int,
    self_dim: int,
    rollout_horizon: int,
    n_rollouts: int,
    n_warmup_steps: int,
    goal_collect_max_steps: int,
    c1_variance_ratio: float,
    c2_score_ratio: float,
    c3_pos_frac_threshold: float,
) -> Dict:
    device = torch.device("cpu")
    torch.manual_seed(seed)
    random.seed(seed)

    print(
        f"[EXQ-104] rollout_horizon={rollout_horizon}, "
        f"n_rollouts={n_rollouts}, n_warmup={n_warmup_steps}",
        flush=True,
    )

    agent, env = _build_agent(seed, world_dim, self_dim)
    agent.eval()

    # --- Phase 1: get goal template from resource contact ---
    print("[EXQ-104] Collecting goal template (resource contact)...", flush=True)
    z_goal_template = _collect_goal_template(agent, env, seed, goal_collect_max_steps)

    # Force-seed GoalState
    goal_config = GoalConfig(
        goal_dim=world_dim,
        z_goal_enabled=True,
        goal_weight=1.0,
    )
    goal_state = GoalState(goal_config, device)
    goal_state._z_goal = z_goal_template.to(device)
    print(
        f"  z_goal seeded: norm={goal_state.goal_norm():.4f}",
        flush=True,
    )

    # --- Phase 2: generate rollouts under 3 conditions ---
    print("[EXQ-104] Generating rollouts (FROZEN)...", flush=True)
    trajs_A = _generate_rollouts_frozen(
        agent, env, seed, rollout_horizon, n_rollouts, n_warmup_steps
    )

    print("[EXQ-104] Generating rollouts (E2_WORLD)...", flush=True)
    trajs_B = _generate_rollouts_e2_world(
        agent, env, seed, rollout_horizon, n_rollouts, n_warmup_steps
    )

    print("[EXQ-104] Generating rollouts (E1_PRED)...", flush=True)
    trajs_C = _generate_rollouts_e1_pred(
        agent, env, seed, rollout_horizon, n_rollouts, n_warmup_steps
    )

    # --- Phase 3: score all conditions ---
    e3 = agent.e3
    metrics_A = _score_trajectories(trajs_A, goal_state, e3)
    metrics_B = _score_trajectories(trajs_B, goal_state, e3)
    metrics_C = _score_trajectories(trajs_C, goal_state, e3)

    print(f"  FROZEN:   mean={metrics_A['goal_score_mean']:.4f}  var={metrics_A['goal_score_var']:.6f}  pos_frac={metrics_A['prox_delta_pos_frac']:.3f}", flush=True)
    print(f"  E2_WORLD: mean={metrics_B['goal_score_mean']:.4f}  var={metrics_B['goal_score_var']:.6f}  pos_frac={metrics_B['prox_delta_pos_frac']:.3f}", flush=True)
    print(f"  E1_PRED:  mean={metrics_C['goal_score_mean']:.4f}  var={metrics_C['goal_score_var']:.6f}  pos_frac={metrics_C['prox_delta_pos_frac']:.3f}", flush=True)

    # Best parallel condition (max variance)
    best_parallel_var   = max(metrics_B["goal_score_var"], metrics_C["goal_score_var"])
    best_parallel_score = max(metrics_B["goal_score_mean"], metrics_C["goal_score_mean"])
    best_parallel_pos   = max(metrics_B["prox_delta_pos_frac"], metrics_C["prox_delta_pos_frac"])
    frozen_var          = max(metrics_A["goal_score_var"], 1e-9)   # avoid div/0

    var_ratio   = best_parallel_var / frozen_var
    score_ratio = best_parallel_score / max(metrics_A["goal_score_mean"], 1e-9)

    # Pass criteria
    c1 = var_ratio >= c1_variance_ratio
    c2 = score_ratio >= c2_score_ratio
    c3 = best_parallel_pos >= c3_pos_frac_threshold

    print(f"\n[EXQ-104] C1 (var_ratio >= {c1_variance_ratio}): {c1}  (ratio={var_ratio:.2f})", flush=True)
    print(f"[EXQ-104] C2 (score_ratio >= {c2_score_ratio}): {c2}  (ratio={score_ratio:.3f})", flush=True)
    print(f"[EXQ-104] C3 (pos_frac >= {c3_pos_frac_threshold}): {c3}  (frac={best_parallel_pos:.3f})", flush=True)

    result = {
        "experiment_type":   EXPERIMENT_TYPE,
        "claim_ids":         CLAIM_IDS,
        "seed":              seed,
        "rollout_horizon":   rollout_horizon,
        "n_rollouts":        n_rollouts,
        "world_dim":         world_dim,
        "self_dim":          self_dim,
        "z_goal_norm":       goal_state.goal_norm(),
        "frozen_goal_score_mean":   metrics_A["goal_score_mean"],
        "frozen_goal_score_var":    metrics_A["goal_score_var"],
        "frozen_prox_delta_pos":    metrics_A["prox_delta_pos_frac"],
        "e2world_goal_score_mean":  metrics_B["goal_score_mean"],
        "e2world_goal_score_var":   metrics_B["goal_score_var"],
        "e2world_prox_delta_pos":   metrics_B["prox_delta_pos_frac"],
        "e1pred_goal_score_mean":   metrics_C["goal_score_mean"],
        "e1pred_goal_score_var":    metrics_C["goal_score_var"],
        "e1pred_prox_delta_pos":    metrics_C["prox_delta_pos_frac"],
        "variance_ratio":    var_ratio,
        "score_ratio":       score_ratio,
        "best_parallel_pos_frac": best_parallel_pos,
        "c1_variance_ratio_pass": bool(c1),
        "c2_score_ratio_pass":    bool(c2),
        "c3_pos_frac_pass":       bool(c3),
        "criteria_met":   sum([c1, c2, c3]),
        "criteria_total": 3,
        "status": "PASS" if (c1 and c2 and c3) else "FAIL",
    }

    print(f"[EXQ-104] Status: {result['status']}", flush=True)
    return result


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser(
        description="V3-EXQ-104: E1/E2 parallel rollout vs frozen z_world (MECH-135)"
    )
    parser.add_argument("--seed",              type=int,   default=42)
    parser.add_argument("--world-dim",         type=int,   default=32)
    parser.add_argument("--self-dim",          type=int,   default=32)
    parser.add_argument("--rollout-horizon",   type=int,   default=30)
    parser.add_argument("--n-rollouts",        type=int,   default=60)
    parser.add_argument("--warmup-steps",      type=int,   default=20)
    parser.add_argument("--goal-max-steps",    type=int,   default=2000)
    parser.add_argument("--c1-var-ratio",      type=float, default=10.0)
    parser.add_argument("--c2-score-ratio",    type=float, default=1.05)
    parser.add_argument("--c3-pos-frac",       type=float, default=0.40)
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Quick smoke test: 5 rollouts, 10 warmup steps, relaxed criteria.",
    )
    args = parser.parse_args()

    if args.smoke_test:
        n_rollouts     = 5
        warmup_steps   = 10
        goal_max_steps = 300
        c1_ratio = 1.0   # relaxed for smoke
        c2_ratio = 1.0
        c3_frac  = 0.0
        print("[V3-EXQ-104] SMOKE TEST MODE", flush=True)
    else:
        n_rollouts     = args.n_rollouts
        warmup_steps   = args.warmup_steps
        goal_max_steps = args.goal_max_steps
        c1_ratio = args.c1_var_ratio
        c2_ratio = args.c2_score_ratio
        c3_frac  = args.c3_pos_frac

    result = run(
        seed=args.seed,
        world_dim=args.world_dim,
        self_dim=args.self_dim,
        rollout_horizon=args.rollout_horizon,
        n_rollouts=n_rollouts,
        n_warmup_steps=warmup_steps,
        goal_collect_max_steps=goal_max_steps,
        c1_variance_ratio=c1_ratio,
        c2_score_ratio=c2_ratio,
        c3_pos_frac_threshold=c3_frac,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"]      = ts
    result["run_id"]             = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = "ree_hybrid_guardrails_v1"
    result["verdict"]            = result["status"]

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
        print("[V3-EXQ-104] SMOKE TEST COMPLETE", flush=True)
        for k in ["variance_ratio", "score_ratio", "best_parallel_pos_frac", "criteria_met"]:
            print(f"  {k}: {result.get(k, 'N/A')}", flush=True)
