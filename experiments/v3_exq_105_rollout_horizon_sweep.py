"""
V3-EXQ-105 -- Rollout Horizon Sweep x Co-evolution Condition (MECH-135 interaction test)

Claims: MECH-135

MECH-135 asserts: the value of a longer planning rollout depends on whether E1/E2
co-evolve z_world during that rollout. Specifically:
  - Longer rollout WITH co-evolution: better goal detection (more steps to see goal approach)
  - Longer rollout WITH frozen z_world: not better or worse (stale world for more steps)

This experiment tests the horizon x co-evolution interaction with forced goal seeding,
using the E2.world_forward condition confirmed in EXQ-104.

Design:
  3 horizons x 2 conditions = 6 cells.
  Horizons: {10, 20, 30} steps.
  Conditions: FROZEN (z_world held at t=0) vs E2_WORLD (E2.world_forward each step).
  Forced goal seeding from resource-contact z_world.
  40 rollouts per cell (240 total).

Metrics per cell:
  - goal_score_mean: mean E3 cumulative goal proximity score
  - goal_score_var:  variance across rollouts
  - prox_delta: mean change in goal_proximity from step 0 to step N

Pass criteria (interaction test):
  C1: goal_score_mean(h=30, E2_WORLD) > goal_score_mean(h=10, E2_WORLD) * 1.05.
      Longer co-evolved rollout is better (each extra step can detect goal approach).
  C2: goal_score_mean(h=30, FROZEN) <= goal_score_mean(h=10, FROZEN) * 1.10.
      Longer frozen rollout is NOT substantially better (stale world accumulates).
      Note: frozen score may increase slightly due to summing more identical goal-proximity
      terms -- the raw sum grows with horizon even when frozen. C2 tests that the gain
      is <= proportional scaling, which is the informative null for E2_WORLD benefit.
  C3 (interaction): goal_score improvement from h=10->30 is larger in E2_WORLD than FROZEN.
      ratio_E2W = mean(h=30,E2W) / mean(h=10,E2W)
      ratio_FRZ = mean(h=30,FRZ) / mean(h=10,FRZ)
      C3 passes if ratio_E2W > ratio_FRZ * c3_interaction_margin.

FAIL means: parallel rollout provides no additional goal-detection benefit relative to
frozen as rollout horizon increases. This would significantly weaken MECH-135.
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


EXPERIMENT_TYPE = "v3_exq_105_rollout_horizon_sweep"
CLAIM_IDS = ["MECH-135"]


# ---------------------------------------------------------------------------
# Helpers (shared setup with EXQ-104)
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


def _collect_goal_template(
    agent: REEAgent,
    env: CausalGridWorldV2,
    seed: int,
    max_steps: int,
) -> torch.Tensor:
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
        if info.get("transition_type", "none") == "resource":
            return latent.z_world.detach()
        if done:
            _, obs_dict = env.reset()
            agent.reset()
    z_goal = torch.randn(1, agent.config.latent.world_dim)
    return F.normalize(z_goal, dim=-1)


def _get_start_state(
    agent: REEAgent,
    env: CausalGridWorldV2,
    seed: int,
    n_warmup_steps: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed + 2000)
    random.seed(seed + 2000)
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
        obs_body  = obs_dict["body_state"]
        obs_world = obs_dict["world_state"]
        with torch.no_grad():
            latent = agent.sense(obs_body, obs_world)
    z_self  = latent.z_self.detach()
    z_world = latent.z_world.detach()
    return z_self, z_world, torch.cat([z_self, z_world], dim=-1)


# ---------------------------------------------------------------------------
# Rollout generators (frozen and E2_WORLD)
# ---------------------------------------------------------------------------

def _run_cell(
    agent: REEAgent,
    env: CausalGridWorldV2,
    seed: int,
    horizon: int,
    n_rollouts: int,
    n_warmup_steps: int,
    frozen: bool,
    goal_state: GoalState,
) -> Dict[str, float]:
    """
    Run n_rollouts for a single (horizon, frozen/parallel) cell.
    Returns aggregated metrics.
    """
    device = agent.device
    scores:      List[float] = []
    prox_deltas: List[float] = []

    for r in range(n_rollouts):
        z_self, z_world, _ = _get_start_state(agent, env, seed + r * 13, n_warmup_steps)

        z_self_states  = [z_self]
        z_world_states = [z_world]
        actions_list   = []

        z_self_curr  = z_self
        z_world_curr = z_world

        for _ in range(horizon):
            a_idx  = random.randint(0, env.action_dim - 1)
            action = _action_to_onehot(a_idx, env.action_dim, device)
            with torch.no_grad():
                z_self_next = agent.e2.predict_next_self(z_self_curr, action)
                if not frozen:
                    z_world_next = agent.e2.world_forward(z_world_curr, action)
                else:
                    z_world_next = z_world   # frozen: always t=0 world state

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

        with torch.no_grad():
            score = agent.e3.compute_goal_score(traj, goal_state)
        scores.append(float(score.mean().item()))

        if len(traj.world_states) >= 2:
            with torch.no_grad():
                p0 = goal_state.goal_proximity(traj.world_states[0]).mean().item()
                p1 = goal_state.goal_proximity(traj.world_states[-1]).mean().item()
            prox_deltas.append(float(p1 - p0))

    scores_t = torch.tensor(scores)
    return {
        "goal_score_mean":    float(scores_t.mean().item()),
        "goal_score_var":     float(scores_t.var().item()) if len(scores) > 1 else 0.0,
        "prox_delta_mean":    float(sum(prox_deltas) / max(len(prox_deltas), 1)),
        "prox_delta_pos_frac": float(sum(d > 0 for d in prox_deltas) / max(len(prox_deltas), 1)),
    }


# ---------------------------------------------------------------------------
# Main run function
# ---------------------------------------------------------------------------

def run(
    seed: int,
    world_dim: int,
    self_dim: int,
    horizons: List[int],
    n_rollouts_per_cell: int,
    n_warmup_steps: int,
    goal_collect_max_steps: int,
    c1_horizon_gain_e2w: float,
    c2_horizon_cap_frz: float,
    c3_interaction_margin: float,
) -> Dict:
    device = torch.device("cpu")
    torch.manual_seed(seed)
    random.seed(seed)

    print(
        f"[EXQ-105] horizons={horizons}, n_rollouts_per_cell={n_rollouts_per_cell}",
        flush=True,
    )

    agent, env = _build_agent(seed, world_dim, self_dim)
    agent.eval()

    # Goal template
    print("[EXQ-105] Collecting goal template...", flush=True)
    z_goal_template = _collect_goal_template(agent, env, seed, goal_collect_max_steps)
    goal_config = GoalConfig(goal_dim=world_dim, z_goal_enabled=True, goal_weight=1.0)
    goal_state  = GoalState(goal_config, device)
    goal_state._z_goal = z_goal_template.to(device)
    print(f"  z_goal_norm={goal_state.goal_norm():.4f}", flush=True)

    # Run all 6 cells
    cell_results: Dict = {}
    for horizon in horizons:
        for cond_name, frozen in [("FROZEN", True), ("E2_WORLD", False)]:
            key = f"h{horizon}_{cond_name}"
            print(f"[EXQ-105] Cell: {key}...", flush=True)
            metrics = _run_cell(
                agent, env, seed, horizon, n_rollouts_per_cell,
                n_warmup_steps, frozen, goal_state,
            )
            cell_results[key] = metrics
            print(
                f"  mean={metrics['goal_score_mean']:.4f}  "
                f"var={metrics['goal_score_var']:.6f}  "
                f"prox_delta={metrics['prox_delta_mean']:.5f}",
                flush=True,
            )

    h_lo = min(horizons)
    h_hi = max(horizons)
    key_hi_e2w = f"h{h_hi}_E2_WORLD"
    key_lo_e2w = f"h{h_lo}_E2_WORLD"
    key_hi_frz = f"h{h_hi}_FROZEN"
    key_lo_frz = f"h{h_lo}_FROZEN"

    mean_hi_e2w = cell_results[key_hi_e2w]["goal_score_mean"]
    mean_lo_e2w = cell_results[key_lo_e2w]["goal_score_mean"]
    mean_hi_frz = cell_results[key_hi_frz]["goal_score_mean"]
    mean_lo_frz = cell_results[key_lo_frz]["goal_score_mean"]

    ratio_e2w = mean_hi_e2w / max(mean_lo_e2w, 1e-9)
    ratio_frz = mean_hi_frz / max(mean_lo_frz, 1e-9)

    # C1: longer co-evolved rollout is better
    c1 = mean_hi_e2w >= mean_lo_e2w * c1_horizon_gain_e2w
    # C2: longer frozen rollout gain is bounded
    c2 = mean_hi_frz <= mean_lo_frz * c2_horizon_cap_frz
    # C3: interaction -- E2_WORLD horizon gain > FROZEN horizon gain * margin
    c3 = ratio_e2w >= ratio_frz * c3_interaction_margin

    print(f"\n[EXQ-105] C1 (h{h_hi} E2W >= h{h_lo} E2W * {c1_horizon_gain_e2w}): {c1}", flush=True)
    print(f"  mean_hi_e2w={mean_hi_e2w:.4f}  mean_lo_e2w={mean_lo_e2w:.4f}", flush=True)
    print(f"[EXQ-105] C2 (h{h_hi} FRZ <= h{h_lo} FRZ * {c2_horizon_cap_frz}): {c2}", flush=True)
    print(f"  mean_hi_frz={mean_hi_frz:.4f}  mean_lo_frz={mean_lo_frz:.4f}", flush=True)
    print(f"[EXQ-105] C3 (interaction margin={c3_interaction_margin}): {c3}", flush=True)
    print(f"  ratio_e2w={ratio_e2w:.3f}  ratio_frz={ratio_frz:.3f}", flush=True)

    result: Dict = {
        "experiment_type":   EXPERIMENT_TYPE,
        "claim_ids":         CLAIM_IDS,
        "seed":              seed,
        "horizons":          horizons,
        "n_rollouts_per_cell": n_rollouts_per_cell,
        "world_dim":         world_dim,
        "self_dim":          self_dim,
        "z_goal_norm":       goal_state.goal_norm(),
    }

    for key, metrics in cell_results.items():
        for mk, mv in metrics.items():
            result[f"{key}_{mk}"] = mv

    result["ratio_e2w"]          = float(ratio_e2w)
    result["ratio_frz"]          = float(ratio_frz)
    result["c1_horizon_gain_e2w_pass"] = bool(c1)
    result["c2_horizon_cap_frz_pass"]  = bool(c2)
    result["c3_interaction_pass"]      = bool(c3)
    result["criteria_met"]   = sum([c1, c2, c3])
    result["criteria_total"] = 3
    result["status"] = "PASS" if (c1 and c2 and c3) else "FAIL"

    print(f"[EXQ-105] Status: {result['status']}", flush=True)
    return result


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser(
        description="V3-EXQ-105: rollout horizon sweep x co-evolution (MECH-135)"
    )
    parser.add_argument("--seed",              type=int,   default=42)
    parser.add_argument("--world-dim",         type=int,   default=32)
    parser.add_argument("--self-dim",          type=int,   default=32)
    parser.add_argument("--horizons",          type=int,   nargs="+", default=[10, 20, 30])
    parser.add_argument("--n-rollouts",        type=int,   default=40)
    parser.add_argument("--warmup-steps",      type=int,   default=20)
    parser.add_argument("--goal-max-steps",    type=int,   default=2000)
    parser.add_argument("--c1-gain",           type=float, default=1.05)
    parser.add_argument("--c2-cap",            type=float, default=1.10,
                        help="Frozen ratio cap (C2 passes if h_hi <= h_lo * this)")
    parser.add_argument("--c3-margin",         type=float, default=1.05,
                        help="Interaction margin: ratio_E2W must be > ratio_FRZ * this")
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Quick smoke test: horizons=[5,10], 3 rollouts, writes output.",
    )
    args = parser.parse_args()

    if args.smoke_test:
        horizons       = [5, 10]
        n_rollouts     = 3
        warmup_steps   = 5
        goal_max_steps = 300
        c1_gain   = 1.0   # relaxed for smoke
        c2_cap    = 999.0
        c3_margin = 1.0
        print("[V3-EXQ-105] SMOKE TEST MODE", flush=True)
    else:
        horizons       = args.horizons
        n_rollouts     = args.n_rollouts
        warmup_steps   = args.warmup_steps
        goal_max_steps = args.goal_max_steps
        c1_gain   = args.c1_gain
        c2_cap    = args.c2_cap
        c3_margin = args.c3_margin

    result = run(
        seed=args.seed,
        world_dim=args.world_dim,
        self_dim=args.self_dim,
        horizons=horizons,
        n_rollouts_per_cell=n_rollouts,
        n_warmup_steps=warmup_steps,
        goal_collect_max_steps=goal_max_steps,
        c1_horizon_gain_e2w=c1_gain,
        c2_horizon_cap_frz=c2_cap,
        c3_interaction_margin=c3_margin,
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
        print("[V3-EXQ-105] SMOKE TEST COMPLETE", flush=True)
        for k in ["ratio_e2w", "ratio_frz", "criteria_met", "status"]:
            print(f"  {k}: {result.get(k, 'N/A')}", flush=True)
