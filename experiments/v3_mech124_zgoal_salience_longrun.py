#!/opt/local/bin/python3
"""
MECH-124 z_goal Salience Long-Run Diagnostic

MECH-124: Consolidation-Mediated Option-Space Contraction (V4 failure mode).
Risk: V4 consolidation amplifies whatever imbalance exists between z_goal salience
and harm salience at the end of V3 training. If harm dominates the residue field
while z_goal signal is weak, SWS replay will strengthen harm representations
further, progressively narrowing the option space the agent considers viable.

This experiment measures whether z_goal salience stays competitive with harm
salience over long runs (500 episodes by default; 1000+ on Spark with --large).

DO NOT QUEUE until EXQ-074b and EXQ-076 have completed and been reviewed.
The results of those experiments are the primary MECH-124 screen. This long-run
experiment provides deeper characterisation if the screen is positive (risk found).

=============================================================================
QUEUING INSTRUCTIONS (when ready):
  1. Confirm EXQ-074b and EXQ-076 have been reviewed.
  2. Assign an EXQ number (next available after current queue).
  3. For default scale (world_dim=32): machine_affinity="any", ~140 min
  4. For Spark scale (--large, world_dim=128): machine_affinity=<spark_hostname>
     Estimated runtime on Spark: ~30 min.
=============================================================================

Experimental design:
  Two conditions:
    BASELINE -- agent with z_goal_enabled + benefit_eval -- no goal focus
    WANTING  -- same, with goal_weight=2.0 (stronger goal drive)

  At every CHECKPOINT_EVERY episodes, record:
    z_goal_norm     -- mean ||z_goal|| per step
    harm_salience   -- mean E3.harm_eval(z_world) per step
    resource_rate   -- resource events / steps
    harm_rate       -- harm events / steps
    ratio           -- z_goal_norm / (harm_salience + eps)

Pass criteria:
  C1: ratio (baseline) > 0.3 at final checkpoint (not harm-dominated)
  C2: ratio (baseline) slope > -0.002/episode over full run (not declining)
  C3: resource_rate positive trend in at least one condition (learning occurs)
  C4: ratio (wanting) >= ratio (baseline) -- stronger goal drive helps

FAIL pattern = MECH-124 risk:
  ratio < 0.2 at final checkpoint, AND/OR ratio declines monotonically.
  Action: flag before V4 consolidation experiments; add balanced replay
  scheduling (MECH-121 guard).
"""

import sys
import argparse
import random
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_mech124_zgoal_salience_longrun"
CLAIM_IDS = ["MECH-124"]

# Default scale (world_dim=32, runs on current hardware)
DEFAULT_EPISODES    = 500
DEFAULT_WORLD_DIM   = 32

# Spark scale (world_dim=128, use --large flag)
LARGE_EPISODES      = 1000
LARGE_WORLD_DIM     = 128

CHECKPOINT_EVERY    = 50    # record metrics every N episodes
STEPS_PER_EP        = 200
SEED                = 42
EPS                 = 1e-6  # prevent division by zero in ratio


def _make_env(seed: int, size: int = 10) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=size,
        num_hazards=3,
        num_resources=5,
        hazard_harm=0.02,
        env_drift_interval=10,
        env_drift_prob=0.05,
        proximity_harm_scale=0.05,
        proximity_benefit_scale=0.03,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
    )


def _make_config(env: CausalGridWorldV2, world_dim: int, goal_weight: float) -> REEConfig:
    if world_dim >= 128:
        config = REEConfig.large(
            body_obs_dim=env.body_obs_dim,
            world_obs_dim=env.world_obs_dim,
            action_dim=env.action_dim,
            reafference_action_dim=env.action_dim,
            z_goal_enabled=True,
            alpha_goal=0.05,
            decay_goal=0.005,
            benefit_eval_enabled=True,
            benefit_weight=1.0,
            goal_weight=goal_weight,
        )
    else:
        config = REEConfig.from_dims(
            body_obs_dim=env.body_obs_dim,
            world_obs_dim=env.world_obs_dim,
            action_dim=env.action_dim,
            self_dim=world_dim,
            world_dim=world_dim,
            alpha_world=0.9,
            alpha_self=0.3,
            reafference_action_dim=env.action_dim,
            z_goal_enabled=True,
            alpha_goal=0.05,
            decay_goal=0.005,
            benefit_eval_enabled=True,
            benefit_weight=1.0,
            goal_weight=goal_weight,
        )
    return config


def _run_condition(
    label: str,
    seed: int,
    n_episodes: int,
    world_dim: int,
    goal_weight: float,
) -> dict:
    """Run one condition; return checkpoint series and final metrics."""
    torch.manual_seed(seed)
    random.seed(seed)

    env    = _make_env(seed, size=10)
    config = _make_config(env, world_dim, goal_weight)
    agent  = REEAgent(config)
    agent.train()

    e1_opt = optim.Adam(agent.e1.parameters(), lr=1e-4)
    e2_opt = optim.Adam(agent.e2.parameters(), lr=3e-4)
    e3_opt = optim.Adam(
        list(agent.e3.parameters()) + list(agent.latent_stack.parameters()),
        lr=1e-3,
    )

    checkpoints = []
    window_goal_norms   = []
    window_harm_sals    = []
    window_resource     = 0
    window_harm_events  = 0
    window_steps        = 0

    for ep in range(n_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        z_self_prev: Optional[torch.Tensor] = None
        action_prev: Optional[torch.Tensor] = None

        ep_goal_norms  = []
        ep_harm_sals   = []
        ep_resources   = 0
        ep_harm_events = 0
        ep_steps       = 0

        for _ in range(STEPS_PER_EP):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            if isinstance(obs_body, torch.Tensor):
                obs_body  = obs_body.float()
                obs_world = obs_world.float()

            latent = agent.sense(obs_body, obs_world)

            # Measure z_goal salience
            z_goal_val = 0.0
            if hasattr(latent, "z_goal") and latent.z_goal is not None:
                z_goal_val = latent.z_goal.detach().norm().item()
            ep_goal_norms.append(z_goal_val)

            # Measure harm salience via E3.harm_eval
            harm_val = 0.0
            try:
                with torch.no_grad():
                    harm_out = agent.e3.harm_eval(latent.z_world.detach())
                    harm_val = harm_out.item() if harm_out.numel() == 1 else harm_out.mean().item()
            except Exception:
                pass
            ep_harm_sals.append(harm_val)

            if z_self_prev is not None and action_prev is not None:
                agent.record_transition(z_self_prev, action_prev, latent.z_self.detach())

            ticks    = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent) if ticks["e1_tick"]
                else torch.zeros(1, world_dim, device=agent.device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action_vec = agent.select_action(candidates, ticks)
            action_idx = action_vec.argmax().item() if action_vec.dim() > 0 else action_vec.item()
            _, reward, done, _, obs_dict = env.step(action_idx)

            if reward > 0:
                ep_resources += 1
            elif reward < 0:
                ep_harm_events += 1

            z_self_prev = latent.z_self.detach()
            action_prev = action_vec.detach()
            ep_steps   += 1
            if done:
                break

        window_goal_norms.extend(ep_goal_norms)
        window_harm_sals.extend(ep_harm_sals)
        window_resource   += ep_resources
        window_harm_events += ep_harm_events
        window_steps       += ep_steps

        # Checkpoint
        if (ep + 1) % CHECKPOINT_EVERY == 0:
            mean_goal = sum(window_goal_norms) / len(window_goal_norms) if window_goal_norms else 0.0
            mean_harm = sum(window_harm_sals)  / len(window_harm_sals)  if window_harm_sals  else 0.0
            ratio     = mean_goal / (mean_harm + EPS)
            res_rate  = window_resource   / window_steps if window_steps > 0 else 0.0
            harm_rate = window_harm_events / window_steps if window_steps > 0 else 0.0

            checkpoints.append({
                "episode":       ep + 1,
                "z_goal_norm":   round(mean_goal, 4),
                "harm_salience": round(mean_harm, 4),
                "ratio":         round(ratio, 4),
                "resource_rate": round(res_rate, 4),
                "harm_rate":     round(harm_rate, 4),
            })

            print(
                f"  [{label}] ep {ep+1:>5}  goal={mean_goal:.3f}"
                f"  harm={mean_harm:.3f}  ratio={ratio:.3f}"
                f"  res={res_rate:.3f}  harm_rate={harm_rate:.3f}",
                flush=True,
            )

            # Reset window
            window_goal_norms   = []
            window_harm_sals    = []
            window_resource     = 0
            window_harm_events  = 0
            window_steps        = 0

    # Compute trend slope (linear regression over ratio series)
    ratios  = [c["ratio"] for c in checkpoints]
    n_pts   = len(ratios)
    slope   = 0.0
    if n_pts >= 2:
        x_mean = (n_pts - 1) / 2.0
        y_mean = sum(ratios) / n_pts
        num    = sum((i - x_mean) * (r - y_mean) for i, r in enumerate(ratios))
        denom  = sum((i - x_mean) ** 2 for i in range(n_pts))
        slope  = num / denom if denom != 0 else 0.0

    final_ratio = ratios[-1] if ratios else 0.0

    return {
        "label":        label,
        "goal_weight":  goal_weight,
        "n_episodes":   n_episodes,
        "world_dim":    world_dim,
        "checkpoints":  checkpoints,
        "final_ratio":  round(final_ratio, 4),
        "ratio_slope":  round(slope, 6),
        "final_resource_rate": checkpoints[-1]["resource_rate"] if checkpoints else 0.0,
        "final_harm_rate":     checkpoints[-1]["harm_rate"]     if checkpoints else 0.0,
    }


if __name__ == "__main__":
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--large", action="store_true",
                        help="Use REEConfig.large() (world_dim=128) for Spark hardware")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    world_dim  = LARGE_WORLD_DIM   if args.large else DEFAULT_WORLD_DIM
    n_episodes = LARGE_EPISODES    if args.large else DEFAULT_EPISODES

    print(f"\n[MECH-124] world_dim={world_dim}  n_episodes={n_episodes}  seed={args.seed}", flush=True)
    print("[MECH-124] Measuring z_goal salience vs harm salience over long run", flush=True)
    print(f"[MECH-124] Checkpoints every {CHECKPOINT_EVERY} episodes", flush=True)

    print("\n[MECH-124] Condition: BASELINE (goal_weight=1.0)", flush=True)
    baseline = _run_condition("BASELINE", args.seed,     n_episodes, world_dim, goal_weight=1.0)

    print("\n[MECH-124] Condition: WANTING (goal_weight=2.0)", flush=True)
    wanting  = _run_condition("WANTING",  args.seed + 1, n_episodes, world_dim, goal_weight=2.0)

    # Pass criteria
    c1 = baseline["final_ratio"] > 0.3
    c2 = baseline["ratio_slope"] > -0.002
    c3 = (baseline["final_resource_rate"] > 0.01
          or wanting["final_resource_rate"] > 0.01)
    c4 = wanting["final_ratio"] >= baseline["final_ratio"] - 0.05  # allowing small margin

    mech124_risk = not c1 or not c2
    status = "PASS" if (c1 and c2 and c3) else "FAIL"

    print(f"\n[MECH-124] -- Results ------------------------------------------", flush=True)
    print(f"  C1 final_ratio>0.3 (not harm-dominated): {c1}  ({baseline['final_ratio']:.3f})", flush=True)
    print(f"  C2 ratio_slope>-0.002 (not declining):   {c2}  ({baseline['ratio_slope']:.5f}/ep)", flush=True)
    print(f"  C3 resource_rate positive trend:         {c3}", flush=True)
    print(f"  C4 wanting >= baseline ratio:            {c4}", flush=True)
    print(f"  MECH-124 V4 risk detected: {mech124_risk}", flush=True)
    print(f"  Status: {status}", flush=True)

    if mech124_risk:
        print("\n[MECH-124] WARNING: risk pattern detected.", flush=True)
        print("  z_goal is not maintaining competitive salience with harm.", flush=True)
        print("  V4 consolidation (MECH-121) will amplify this imbalance.", flush=True)
        print("  Recommended: add balanced replay scheduling in MECH-121 design.", flush=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result = {
        "status": status,
        "metrics": {
            "scale":    "large" if args.large else "default",
            "world_dim": world_dim,
            "baseline": baseline,
            "wanting":  wanting,
            "mech124_risk": mech124_risk,
        },
        "criteria": {
            "C1_ratio_above_threshold": c1,
            "C2_ratio_not_declining":   c2,
            "C3_learning_occurs":       c3,
            "C4_goal_weight_helps":     c4,
        },
        "run_timestamp":      ts,
        "run_id":             f"{EXPERIMENT_TYPE}_{ts}_v3",
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim":              "MECH-124",
        "verdict":            status,
    }

    out_dir = (
        Path(__file__).resolve().parents[2]
        / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{EXPERIMENT_TYPE}_{ts}.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"\nResult written to: {out_path}", flush=True)
    print(f"Status: {result['status']}", flush=True)
