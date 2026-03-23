#!/opt/local/bin/python3
"""
Spark-Scale SD-005 Validation: z_self/z_world Split vs Unified Latent

SD-005 asserts that splitting the latent into z_self (proprioceptive, E2 domain)
and z_world (exteroceptive, E3/Hippocampal domain) improves specialisation of each
module relative to a unified representation. This was validated at world_dim=32 in
EXQ-044. This experiment replicates the key SD-005 ablation at REEConfig.large()
(world_dim=128) and optionally REEConfig.xlarge() (world_dim=256).

At small scale (world_dim=32) both conditions may perform similarly because the
network has sufficient capacity to partition implicitly even without the structural
split. At large scale, the split provides explicit routing that the network cannot
easily replicate, making any performance difference more pronounced.

Two conditions (matched seeds):
  SPLIT   -- REEConfig.large() default (unified_latent_mode=False)
  UNIFIED -- same config with unified_latent_mode=True

Metrics (split - unified at convergence):
  delta_e3_harm_discrim  -- split E3 harm discrimination gap minus unified gap
  delta_harm_rate        -- harm_rate_unified - harm_rate_split (positive = split wins)
  delta_resource_rate    -- resource_rate_split - resource_rate_unified
  e2_pred_error_split    -- E2 prediction error (should be lower with split)
  e2_pred_error_unified

PASS (ALL required):
  C1: delta_e3_harm_discrim > 0.02 (split E3 discriminates harm better at scale)
  C2: delta_harm_rate > 0.01 (split reduces harm)
  C3: e2_pred_error_split <= e2_pred_error_unified (split E2 no worse)
  C4: resource_rate_split >= resource_rate_unified - 0.02 (split does not hurt approach)

=============================================================================
QUEUING: Do NOT queue until Spark hardware is available.
  Use --large (world_dim=128, 1 Spark) or --xlarge (world_dim=256, 2 Sparks).
  Assign next available EXQ number when queuing.
  Estimated runtime (Spark, world_dim=128): ~25 min
  Estimated runtime (Spark, world_dim=256): ~60 min
=============================================================================
"""

import sys
import argparse
import random
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.optim as optim
import torch.nn.functional as F

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_spark_sd005_scale_validation"
CLAIM_IDS = ["SD-005"]

WARMUP_EPISODES = 300
EVAL_EPISODES   = 50
STEPS_PER_EP    = 200
SEED            = 42


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


def _make_config(env: CausalGridWorldV2, scale: str, unified: bool) -> REEConfig:
    if scale == "xlarge":
        config = REEConfig.xlarge(
            body_obs_dim=env.body_obs_dim,
            world_obs_dim=env.world_obs_dim,
            action_dim=env.action_dim,
            reafference_action_dim=env.action_dim,
        )
    else:  # large (default)
        config = REEConfig.large(
            body_obs_dim=env.body_obs_dim,
            world_obs_dim=env.world_obs_dim,
            action_dim=env.action_dim,
            reafference_action_dim=env.action_dim,
        )
    if unified:
        config.latent.unified_latent_mode = True
    return config


def _run_condition(label: str, seed: int, scale: str, unified: bool) -> dict:
    torch.manual_seed(seed)
    random.seed(seed)

    env    = _make_env(seed)
    config = _make_config(env, scale, unified)
    world_dim = config.latent.world_dim
    agent  = REEAgent(config)

    e1_opt = optim.Adam(agent.e1.parameters(), lr=1e-4)
    e2_opt = optim.Adam(agent.e2.parameters(), lr=3e-4)
    e3_opt = optim.Adam(
        list(agent.e3.parameters()) + list(agent.latent_stack.parameters()),
        lr=1e-3,
    )
    agent.train()

    # -- Warmup training -------------------------------------------------------
    print(f"  [{label}] warmup ({WARMUP_EPISODES} eps) world_dim={world_dim}", flush=True)
    for ep in range(WARMUP_EPISODES):
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
            print(f"  [{label}] warmup ep {ep+1}/{WARMUP_EPISODES}", flush=True)

    # -- Eval ------------------------------------------------------------------
    print(f"  [{label}] eval ({EVAL_EPISODES} eps)", flush=True)
    agent.eval()

    harm_events    = 0
    resource_visits = 0
    total_steps    = 0

    e3_harm_on_harm:  list = []  # E3.harm_eval output on actual harm steps
    e3_harm_on_safe:  list = []  # E3.harm_eval output on safe steps
    e2_losses:        list = []

    with torch.no_grad():
        for ep in range(EVAL_EPISODES):
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
                harm_signal = float(reward) if reward < 0 else 0.0
                ttype = info.get("transition_type", "none")

                if ttype in ("agent_caused_hazard", "hazard_approach"):
                    harm_events += 1
                if ttype == "resource":
                    resource_visits += 1

                # Measure E3 harm discrimination
                harm_pred = agent.e3.harm_eval(latent.z_world.detach()).item()
                if harm_signal < 0:
                    e3_harm_on_harm.append(harm_pred)
                else:
                    e3_harm_on_safe.append(harm_pred)

                # E2 prediction quality (no grad needed, already in no_grad block)
                try:
                    e2_l = agent.compute_e2_loss()
                    e2_losses.append(e2_l.item())
                except Exception:
                    pass

                total_steps += 1
                if done:
                    break

    harm_rate      = harm_events / max(1, total_steps)
    resource_rate  = resource_visits / max(1, total_steps)
    mean_e3_harm   = sum(e3_harm_on_harm) / max(1, len(e3_harm_on_harm))
    mean_e3_safe   = sum(e3_harm_on_safe) / max(1, len(e3_harm_on_safe))
    e3_discrim_gap = mean_e3_harm - mean_e3_safe   # positive = E3 distinguishes harm
    mean_e2_loss   = sum(e2_losses) / max(1, len(e2_losses))

    print(
        f"  [{label}] harm_rate={harm_rate:.4f}"
        f"  resource_rate={resource_rate:.4f}"
        f"  e3_discrim={e3_discrim_gap:.4f}"
        f"  e2_loss={mean_e2_loss:.5f}",
        flush=True,
    )

    return {
        "label":           label,
        "unified":         unified,
        "world_dim":       world_dim,
        "harm_rate":       round(harm_rate, 5),
        "resource_rate":   round(resource_rate, 5),
        "e3_discrim_gap":  round(e3_discrim_gap, 5),
        "mean_e3_on_harm": round(mean_e3_harm, 5),
        "mean_e3_on_safe": round(mean_e3_safe, 5),
        "e2_pred_error":   round(mean_e2_loss, 6),
        "n_harm_steps":    len(e3_harm_on_harm),
        "n_safe_steps":    len(e3_harm_on_safe),
    }


if __name__ == "__main__":
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",   type=int, default=SEED)
    parser.add_argument("--xlarge", action="store_true", help="Use world_dim=256 (two Sparks)")
    args = parser.parse_args()

    scale = "xlarge" if args.xlarge else "large"
    print(f"\n[SD-005-SCALE] scale={scale}  seed={args.seed}", flush=True)
    print("[SD-005-SCALE] Condition: SPLIT (z_self/z_world separate)", flush=True)
    r_split   = _run_condition("SPLIT",   args.seed,     scale, unified=False)
    print("[SD-005-SCALE] Condition: UNIFIED (unified_latent_mode=True)", flush=True)
    r_unified = _run_condition("UNIFIED", args.seed + 1, scale, unified=True)

    delta_discrim   = r_split["e3_discrim_gap"] - r_unified["e3_discrim_gap"]
    delta_harm      = r_unified["harm_rate"]    - r_split["harm_rate"]
    delta_resource  = r_split["resource_rate"]  - r_unified["resource_rate"]

    c1 = delta_discrim  > 0.02
    c2 = delta_harm     > 0.01
    c3 = r_split["e2_pred_error"] <= r_unified["e2_pred_error"]
    c4 = delta_resource > -0.02

    status = "PASS" if (c1 and c2 and c3 and c4) else "FAIL"

    print(f"\n[SD-005-SCALE] -- Results -------------------------------------------", flush=True)
    print(f"  delta_e3_harm_discrim: {delta_discrim:.4f}  (C1 > 0.02: {c1})", flush=True)
    print(f"  delta_harm_rate:       {delta_harm:.4f}  (C2 > 0.01: {c2})", flush=True)
    print(f"  e2_loss split/unified: {r_split['e2_pred_error']:.5f} / {r_unified['e2_pred_error']:.5f}  (C3: {c3})", flush=True)
    print(f"  delta_resource_rate:   {delta_resource:.4f}  (C4 > -0.02: {c4})", flush=True)
    print(f"  Status: {status}", flush=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result = {
        "status": status,
        "metrics": {
            "scale": scale,
            "split":   r_split,
            "unified": r_unified,
            "delta_e3_harm_discrim": round(delta_discrim, 5),
            "delta_harm_rate":       round(delta_harm, 5),
            "delta_resource_rate":   round(delta_resource, 5),
        },
        "criteria": {
            "C1_discrim_gap": c1,
            "C2_harm_rate":   c2,
            "C3_e2_loss":     c3,
            "C4_resource":    c4,
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
