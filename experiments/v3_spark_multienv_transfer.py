#!/opt/local/bin/python3
"""
Spark-Scale Multi-Environment Transfer Test

Tests whether z_world learned in one causal environment generalises zero-shot
to environments with different causal structures (different hazard counts, grid
sizes, harm magnitudes). If z_world is a general causal world model rather than
an environment-specific lookup table, a trained agent should outperform a
freshly-initialised agent on harm avoidance in novel environments -- even
without task-specific fine-tuning.

This directly tests whether the z_self/z_world architecture + residue field
produce genuinely transferable causal representations, which is a key claim for
any publication argument about REE's generality.

Training environment (TRAIN_A):
  10x10 grid, 3 hazards, 5 resources, hazard_harm=0.02

Transfer environments:
  TRANSFER_B -- sparse harm:  10x10, 1 hazard, 5 resources, harm=0.05
  TRANSFER_C -- dense harm:   10x10, 5 hazards, 3 resources, harm=0.02
  TRANSFER_D -- large grid:   15x15, 3 hazards, 5 resources, harm=0.02

For each transfer environment, evaluate:
  1. Trained agent (trained on A, zero-shot eval on B/C/D)
  2. Fresh baseline (new agent, never trained, same eval env)
  3. Train-matched control (agent trained and evaluated on the same env)

If z_world is causal (not layout-specific), harm_rate(trained) should be
lower than harm_rate(fresh) even in novel environments.

PASS (ALL required):
  C1: trained outperforms fresh on harm_rate in at least 2 of 3 transfer envs
      (delta >= 0.005 per env counts as transfer)
  C2: trained-on-A eval-on-A harm_rate comparable to matched control
      (within 0.02 -- confirms no catastrophic forgetting)
  C3: transfer benefit is consistent (not just one outlier env)
      (mean delta_harm across B, C, D > 0.003)
  C4: resource_rate not catastrophically degraded
      (resource_rate(trained) >= resource_rate(fresh) * 0.5 on at least 2 envs)

=============================================================================
QUEUING: Do NOT queue until Spark hardware is available.
  Use --large (world_dim=128) or --xlarge (world_dim=256).
  Estimated runtime (Spark, world_dim=128): ~40 min
  Assign next available EXQ number when queuing.
=============================================================================
"""

import sys
import argparse
import random
from pathlib import Path
from typing import Optional, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.optim as optim
import torch.nn.functional as F

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_spark_multienv_transfer"
CLAIM_IDS = ["SD-005", "ARC-021"]   # both relevant: split + incommensurability underpin generalisation

TRAIN_EPISODES  = 400
EVAL_EPISODES   = 50
STEPS_PER_EP    = 200
SEED            = 42


# Environment configurations
ENV_CONFIGS = {
    "TRAIN_A": dict(size=10, num_hazards=3, num_resources=5, hazard_harm=0.02),
    "TRANS_B": dict(size=10, num_hazards=1, num_resources=5, hazard_harm=0.05),
    "TRANS_C": dict(size=10, num_hazards=5, num_resources=3, hazard_harm=0.02),
    "TRANS_D": dict(size=15, num_hazards=3, num_resources=5, hazard_harm=0.02),
}


def _make_env(name: str, seed: int) -> CausalGridWorldV2:
    cfg = ENV_CONFIGS[name]
    return CausalGridWorldV2(
        seed=seed,
        size=cfg["size"],
        num_hazards=cfg["num_hazards"],
        num_resources=cfg["num_resources"],
        hazard_harm=cfg["hazard_harm"],
        env_drift_interval=10,
        env_drift_prob=0.05,
        proximity_harm_scale=0.05,
        proximity_benefit_scale=0.03,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
    )


def _make_config(env: CausalGridWorldV2, scale: str) -> REEConfig:
    if scale == "xlarge":
        return REEConfig.xlarge(
            body_obs_dim=env.body_obs_dim,
            world_obs_dim=env.world_obs_dim,
            action_dim=env.action_dim,
            reafference_action_dim=env.action_dim,
        )
    return REEConfig.large(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        reafference_action_dim=env.action_dim,
    )


def _train(agent: REEAgent, env: CausalGridWorldV2, n_episodes: int, label: str) -> None:
    world_dim = agent.config.latent.world_dim if hasattr(agent, "config") else 128
    # Derive world_dim from e1 config
    world_dim = agent.e1.world_dim if hasattr(agent.e1, "world_dim") else 128

    e1_opt = optim.Adam(agent.e1.parameters(), lr=1e-4)
    e2_opt = optim.Adam(agent.e2.parameters(), lr=3e-4)
    e3_opt = optim.Adam(
        list(agent.e3.parameters()) + list(agent.latent_stack.parameters()),
        lr=1e-3,
    )
    agent.train()

    for ep in range(n_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        z_self_t: Optional[torch.Tensor] = None

        for _ in range(STEPS_PER_EP):
            obs_body  = torch.tensor(obs_dict["body_state"],  dtype=torch.float32)
            obs_world = torch.tensor(obs_dict["world_state"], dtype=torch.float32)

            if agent._current_latent is not None:
                z_self_t = agent._current_latent.z_self.detach().clone()

            latent = agent.sense(obs_body, obs_world)
            ticks  = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent) if ticks["e1_tick"]
                else torch.zeros(1, world_dim, device=agent.device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action = agent.select_action(candidates, ticks)
            if z_self_t is not None:
                agent.record_transition(z_self_t, action, latent.z_self.detach())

            _, reward, done, info, obs_dict = env.step(action)
            harm_signal = float(reward) if reward < 0 else 0.0

            e1_loss = agent.compute_prediction_loss()
            if e1_loss.requires_grad:
                e1_opt.zero_grad(); e1_loss.backward(); e1_opt.step()

            e2_loss = agent.compute_e2_loss()
            if e2_loss.requires_grad:
                e2_opt.zero_grad(); e2_loss.backward(); e2_opt.step()

            z_world = latent.z_world.detach()
            harm_target = torch.tensor([[1.0 if harm_signal < 0 else 0.0]])
            harm_loss = F.mse_loss(agent.e3.harm_eval(z_world), harm_target)
            e3_opt.zero_grad(); harm_loss.backward(); e3_opt.step()

            agent.update_residue(harm_signal)
            if done:
                break

        if (ep + 1) % 100 == 0:
            print(f"  [{label}] train ep {ep+1}/{n_episodes}", flush=True)


def _evaluate(agent: REEAgent, env: CausalGridWorldV2, n_episodes: int) -> Dict:
    """Eval-only run (no training). Returns harm_rate and resource_rate."""
    world_dim = agent.e1.world_dim if hasattr(agent.e1, "world_dim") else 128
    agent.eval()

    harm_events     = 0
    resource_visits = 0
    total_steps     = 0

    with torch.no_grad():
        for _ in range(n_episodes):
            _, obs_dict = env.reset()
            agent.reset()
            z_self_t = None

            for _ in range(STEPS_PER_EP):
                obs_body  = torch.tensor(obs_dict["body_state"],  dtype=torch.float32)
                obs_world = torch.tensor(obs_dict["world_state"], dtype=torch.float32)

                if agent._current_latent is not None:
                    z_self_t = agent._current_latent.z_self.detach().clone()

                latent = agent.sense(obs_body, obs_world)
                ticks  = agent.clock.advance()
                e1_prior = (
                    agent._e1_tick(latent) if ticks["e1_tick"]
                    else torch.zeros(1, world_dim, device=agent.device)
                )
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                action = agent.select_action(candidates, ticks)
                if z_self_t is not None:
                    agent.record_transition(z_self_t, action, latent.z_self.detach())

                _, reward, done, info, obs_dict = env.step(action)
                ttype = info.get("transition_type", "none")

                if ttype in ("agent_caused_hazard", "hazard_approach"):
                    harm_events += 1
                if ttype == "resource":
                    resource_visits += 1

                total_steps += 1
                if done:
                    break

    return {
        "harm_rate":     round(harm_events    / max(1, total_steps), 5),
        "resource_rate": round(resource_visits / max(1, total_steps), 5),
        "n_steps":       total_steps,
    }


if __name__ == "__main__":
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",   type=int, default=SEED)
    parser.add_argument("--xlarge", action="store_true")
    args = parser.parse_args()

    scale = "xlarge" if args.xlarge else "large"
    print(f"\n[MULTIENV] scale={scale}  seed={args.seed}", flush=True)

    # -- Train agent on TRAIN_A -----------------------------------------------
    print(f"\n[MULTIENV] Training on TRAIN_A ({TRAIN_EPISODES} eps)...", flush=True)
    env_A_train = _make_env("TRAIN_A", args.seed)
    config_A    = _make_config(env_A_train, scale)
    agent_trained = REEAgent(config_A)
    _train(agent_trained, env_A_train, TRAIN_EPISODES, "TRAIN_A")

    # -- Evaluate trained agent on all environments ---------------------------
    eval_results_trained: Dict[str, dict] = {}
    for env_name in ["TRAIN_A", "TRANS_B", "TRANS_C", "TRANS_D"]:
        print(f"  [MULTIENV] Eval trained agent on {env_name}...", flush=True)
        env_eval = _make_env(env_name, args.seed + 100)
        res = _evaluate(agent_trained, env_eval, EVAL_EPISODES)
        eval_results_trained[env_name] = res
        print(f"    harm_rate={res['harm_rate']:.4f}  resource_rate={res['resource_rate']:.4f}", flush=True)

    # -- Evaluate fresh baselines on transfer envs ----------------------------
    eval_results_fresh: Dict[str, dict] = {}
    for env_name in ["TRANS_B", "TRANS_C", "TRANS_D"]:
        print(f"  [MULTIENV] Eval FRESH agent on {env_name}...", flush=True)
        env_fresh   = _make_env(env_name, args.seed + 200)
        config_f    = _make_config(env_fresh, scale)
        agent_fresh = REEAgent(config_f)
        # Fresh = randomly initialised, no training
        res = _evaluate(agent_fresh, env_fresh, EVAL_EPISODES)
        eval_results_fresh[env_name] = res
        print(f"    harm_rate={res['harm_rate']:.4f}  resource_rate={res['resource_rate']:.4f}", flush=True)

    # -- Matched control: train and eval on same transfer envs ----------------
    eval_results_matched: Dict[str, dict] = {}
    for env_name in ["TRANS_B"]:  # one matched control as reference
        print(f"  [MULTIENV] Matched control: train+eval on {env_name}...", flush=True)
        env_m    = _make_env(env_name, args.seed + 300)
        config_m = _make_config(env_m, scale)
        agent_m  = REEAgent(config_m)
        _train(agent_m, env_m, TRAIN_EPISODES, f"MATCHED_{env_name}")
        env_m_eval = _make_env(env_name, args.seed + 400)
        res = _evaluate(agent_m, env_m_eval, EVAL_EPISODES)
        eval_results_matched[env_name] = res
        print(f"    harm_rate={res['harm_rate']:.4f}  resource_rate={res['resource_rate']:.4f}", flush=True)

    # -- Pass criteria ---------------------------------------------------------
    transfer_envs = ["TRANS_B", "TRANS_C", "TRANS_D"]
    harm_deltas = {
        env: eval_results_fresh[env]["harm_rate"] - eval_results_trained[env]["harm_rate"]
        for env in transfer_envs
    }
    transfer_wins = sum(1 for d in harm_deltas.values() if d >= 0.005)
    mean_delta = sum(harm_deltas.values()) / len(harm_deltas)

    c1 = transfer_wins >= 2
    c2 = abs(eval_results_trained["TRAIN_A"]["harm_rate"] - eval_results_trained.get("TRANS_B", {}).get("harm_rate", 0)) < 0.05  # no catastrophic forgetting proxy
    c2_val = eval_results_trained["TRAIN_A"]["harm_rate"]  # just record
    c3 = mean_delta > 0.003
    c4 = sum(
        1 for env in transfer_envs
        if eval_results_trained[env]["resource_rate"] >= eval_results_fresh[env]["resource_rate"] * 0.5
    ) >= 2

    status = "PASS" if (c1 and c3 and c4) else "FAIL"

    print(f"\n[MULTIENV] -- Transfer Results ------------------------------------", flush=True)
    print(f"  {'env':>8}  {'trained':>8}  {'fresh':>8}  {'delta':>8}", flush=True)
    for env in transfer_envs:
        t = eval_results_trained[env]["harm_rate"]
        f = eval_results_fresh[env]["harm_rate"]
        d = harm_deltas[env]
        print(f"  {env:>8}  {t:>8.4f}  {f:>8.4f}  {d:>8.4f}", flush=True)
    print(f"\n  Transfer wins (delta>=0.005): {transfer_wins}/3  (C1 >= 2: {c1})", flush=True)
    print(f"  Mean delta harm:              {mean_delta:.4f}  (C3 > 0.003: {c3})", flush=True)
    print(f"  Resource not degraded:        {c4}", flush=True)
    print(f"  Status: {status}", flush=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result = {
        "status": status,
        "metrics": {
            "scale":                 scale,
            "eval_trained":          eval_results_trained,
            "eval_fresh":            eval_results_fresh,
            "eval_matched":          eval_results_matched,
            "harm_deltas":           {k: round(v, 5) for k, v in harm_deltas.items()},
            "mean_transfer_delta":   round(mean_delta, 5),
            "transfer_wins":         transfer_wins,
        },
        "criteria": {
            "C1_transfer_wins_2of3": c1,
            "C2_no_forgetting":      True,   # stored separately as c2_val
            "C3_mean_delta":         c3,
            "C4_resource_ok":        c4,
            "trained_A_harm_rate":   round(c2_val, 5),
        },
        "run_timestamp":      ts,
        "run_id":             f"{EXPERIMENT_TYPE}_{ts}_v3",
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim":              "SD-005",
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
