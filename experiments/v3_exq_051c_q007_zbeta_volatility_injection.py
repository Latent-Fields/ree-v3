"""
EXQ-051c -- Q-007: z_beta Volatility Injection (NE/LC analog)

Claim: Q-007 -- z_beta should correlate with environmental volatility
       (running variance of harm prediction errors).

Architectural gap identified in EXQ-051:
  z_beta = beta_encoder(cat(z_self, z_world)) -- purely sensory.
  No pathway from E3's running_variance -> z_beta always ~0.72 in both
  stable and volatile conditions.

Fix tested here (Option A -- Yu & Dayan 2005, PMID 15944135):
  Inject running_variance as explicit scalar input to beta_encoder:
    z_beta = beta_encoder(cat(z_self, z_world, running_variance_scalar))
  Enabled via LatentStackConfig.volatility_signal_dim = 1.

Biological basis:
  LC-NE encodes unexpected uncertainty: NE rises when prediction error
  variance exceeds what was expected under the current contingency.
  This is not a sensory-content signal -- it is a second-order signal
  about HOW UNPREDICTABLE the harm environment has been recently.
  running_variance is the REE analog of HGF log-volatility (mu3), which
  drives affective state in precision-weighted inference models.

Design:
  Two agents trained and evaluated in parallel:
    STABLE:   env_drift_prob=0.01 (hazard layout stable across episodes)
    VOLATILE: env_drift_prob=0.9, env_drift_interval=1 (layout changes each ep)
  Both conditions have identical mean hazard count -> same mean harm rate.
  The difference is PREDICTABILITY -- stable agents learn E3 well
  -> low running_variance; volatile agents keep getting surprised -> high
  running_variance.

  With running_variance injected into beta_encoder:
    -> z_beta should be higher in VOLATILE condition (arousal up)
    -> Pearson r(running_variance, z_beta_norm) > 0.3

Criteria:
  C1: z_beta_norm_volatile > z_beta_norm_stable + 0.05
        (volatile envs produce higher affective arousal)
  C2: Pearson r(running_variance_trajectory, z_beta_norm_trajectory) > 0.3
        (within-agent correlation over training episodes)
  C3: z_beta responds faster to sudden volatility increase (transfer test):
        after 100 stable episodes -> suddenly volatile -> z_beta rises within
        20 episodes (relative to baseline, delta > 0.02)

Fixes vs EXQ-051b:
  - CausalGridWorld called without invalid kwargs (body_obs_dim, world_obs_dim,
    randomise_hazard_positions). Use env_drift_prob=0.9/0.01 instead.
  - WORLD_OBS_DIM: 54 -> 200 (CausalGridWorld, use_proxy_fields=False, size=10)
  - env.reset() API: now returns (_, obs_dict) with obs_dict["body_state"] and
    obs_dict["world_state"]; env.step() returns (_, reward, done, info, obs_dict)
  - Volatility injection: bypass agent.sense() and call body/world encoders then
    latent_stack.encode(..., volatility_signal=rv_tensor) directly, then store
    agent._current_latent = latent.detach()
  - Training loop: use compute_prediction_loss() + compute_e2_loss() pattern with
    combined backward pass; E3 harm supervision via harm_eval(z_world)
  - generate_trajectories/select_action API used correctly
"""

import json
import math
import random
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorld
from ree_core.utils.config import REEConfig

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED             = 42
N_TRAIN_EPISODES = 300
N_EVAL_EPISODES  = 50
TRANSFER_EPISODES = 150

MAX_STEPS    = 150
N_HAZARDS    = 4
GRID_SIZE    = 10
BODY_OBS_DIM = 10   # CausalGridWorld, use_proxy_fields=False
WORLD_OBS_DIM = 200  # CausalGridWorld, size=10, use_proxy_fields=False
ACTION_DIM   = 4
SELF_DIM     = 32
WORLD_DIM    = 32

VOLATILITY_SIGNAL_DIM = 1   # Q-007: scalar running_variance -> beta_encoder

C1_DELTA_THRESHOLD   = 0.05
C2_PEARSON_THRESHOLD = 0.30
C3_DELTA_THRESHOLD   = 0.02
C3_WINDOW            = 20

EXPERIMENT_TYPE = "v3_exq_051c_q007_zbeta_volatility_injection"
CLAIM_IDS = ["Q-007"]

DEVICE = "cpu"

# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_env(volatile: bool, seed: Optional[int] = None) -> CausalGridWorld:
    """Create environment. volatile=True uses high drift to randomise layout."""
    return CausalGridWorld(
        size=GRID_SIZE,
        num_hazards=N_HAZARDS,
        num_resources=3,
        use_proxy_fields=False,
        seed=seed,
        env_drift_prob=0.9 if volatile else 0.01,
        env_drift_interval=1 if volatile else 50,
    )


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def make_agent(seed: int) -> REEAgent:
    config = REEConfig.from_dims(
        body_obs_dim=BODY_OBS_DIM,
        world_obs_dim=WORLD_OBS_DIM,
        action_dim=ACTION_DIM,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        alpha_world=0.9,
        alpha_self=0.3,
    )
    # Q-007: enable volatility injection into beta_encoder
    config.latent.volatility_signal_dim = VOLATILITY_SIGNAL_DIM
    config.e3.commitment_threshold = 0.40

    torch.manual_seed(seed)
    return REEAgent(config)


# ---------------------------------------------------------------------------
# Volatility-injected encode helper
# ---------------------------------------------------------------------------

def encode_with_volatility(agent: REEAgent, obs_body: torch.Tensor,
                            obs_world: torch.Tensor,
                            rv_tensor: torch.Tensor) -> object:
    """
    Bypass agent.sense() to inject volatility_signal into latent_stack.encode().

    agent.sense() does not expose volatility_signal, so we call the encoders
    and latent_stack directly, then update agent._current_latent.
    """
    if obs_body.dim() == 1:
        obs_body = obs_body.unsqueeze(0)
    if obs_world.dim() == 1:
        obs_world = obs_world.unsqueeze(0)

    obs_body  = obs_body.to(agent.device).float()
    obs_world = obs_world.to(agent.device).float()

    enc_body  = agent.body_obs_encoder(obs_body)
    enc_world = agent.world_obs_encoder(obs_world)
    enc_combined = torch.cat([enc_body, enc_world], dim=-1)

    latent = agent.latent_stack.encode(
        enc_combined,
        prev_state=agent._current_latent,
        prev_action=agent._last_action,
        volatility_signal=rv_tensor.to(agent.device),
    )
    agent._current_latent = latent.detach()
    return latent


# ---------------------------------------------------------------------------
# Pearson correlation
# ---------------------------------------------------------------------------

def pearson_r(x: List[float], y: List[float]) -> float:
    if len(x) < 5:
        return 0.0
    xn = np.array(x)
    yn = np.array(y)
    if xn.std() < 1e-9 or yn.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(xn, yn)[0, 1])


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(
    agent: REEAgent,
    env: CausalGridWorld,
    train: bool = True,
    e1_opt: Optional[object] = None,
    e2_opt: Optional[object] = None,
    e3_opt: Optional[object] = None,
) -> dict:
    """Run one episode. Returns per-episode metrics."""
    _, obs_dict = env.reset()
    agent.reset()

    z_beta_norms: List[float] = []
    running_vars: List[float] = []
    total_harm = 0.0

    for _step in range(MAX_STEPS):
        obs_body  = torch.tensor(obs_dict["body_state"],  dtype=torch.float32)
        obs_world = torch.tensor(obs_dict["world_state"], dtype=torch.float32)

        # Q-007: get current running_variance from E3 as volatility signal
        rv = float(agent.e3._running_variance)
        rv_tensor = torch.tensor([[rv]], dtype=torch.float32)

        if train:
            latent = encode_with_volatility(agent, obs_body, obs_world, rv_tensor)
        else:
            with torch.no_grad():
                latent = encode_with_volatility(agent, obs_body, obs_world, rv_tensor)

        z_beta_norms.append(float(latent.z_beta.norm().item()))
        running_vars.append(rv)

        ticks = agent.clock.advance()
        e1_prior = (
            agent._e1_tick(latent) if ticks["e1_tick"]
            else torch.zeros(1, WORLD_DIM, device=agent.device)
        )

        if train:
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action = agent.select_action(candidates, ticks, temperature=1.0)
        else:
            with torch.no_grad():
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                action = agent.select_action(candidates, ticks, temperature=0.5)

        if train and e1_opt is not None:
            # Combined E1 + E2 backward (avoids inplace-op conflicts)
            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            if e1_loss.requires_grad or e2_loss.requires_grad:
                total_loss = e1_loss + e2_loss
                e1_opt.zero_grad()
                e2_opt.zero_grad()
                total_loss.backward()
                e1_opt.step()
                e2_opt.step()

        _, reward, done, info, obs_dict = env.step(action)
        harm_signal = float(reward) if reward < 0 else 0.0
        total_harm += abs(harm_signal)

        # E3 harm supervision
        if train and e3_opt is not None and agent._current_latent is not None:
            z_world = agent._current_latent.z_world.detach()
            harm_target = torch.tensor(
                [[1.0 if harm_signal < 0 else 0.0]], device=agent.device
            )
            harm_loss = F.mse_loss(agent.e3.harm_eval(z_world), harm_target)
            e3_opt.zero_grad()
            harm_loss.backward()
            e3_opt.step()

        agent.update_residue(harm_signal)

        if done:
            break

    return {
        "total_harm": total_harm,
        "mean_z_beta_norm": float(np.mean(z_beta_norms)) if z_beta_norms else 0.0,
        "mean_running_variance": float(np.mean(running_vars)) if running_vars else 0.0,
        "final_running_variance": running_vars[-1] if running_vars else 0.0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(**kwargs) -> dict:
    seed = kwargs.get("seed", SEED)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print("=" * 60, flush=True)
    print("EXQ-051c: Q-007 z_beta Volatility Injection (NE/LC analog)", flush=True)
    print("=" * 60, flush=True)

    lr = kwargs.get("lr", 1e-3)
    n_train = kwargs.get("n_train_episodes", N_TRAIN_EPISODES)
    n_eval  = kwargs.get("n_eval_episodes",  N_EVAL_EPISODES)
    n_transfer = kwargs.get("n_transfer_episodes", TRANSFER_EPISODES)

    agent_stable   = make_agent(seed)
    env_stable     = make_env(volatile=False, seed=seed)
    agent_volatile = make_agent(seed + 1)
    env_volatile   = make_env(volatile=True,  seed=seed + 1)

    e1_opt_s = optim.Adam(agent_stable.e1.parameters(), lr=lr)
    e2_opt_s = optim.Adam(agent_stable.e2.parameters(), lr=lr * 3)
    e3_opt_s = optim.Adam(
        list(agent_stable.e3.parameters()) + list(agent_stable.latent_stack.parameters()),
        lr=lr,
    )

    e1_opt_v = optim.Adam(agent_volatile.e1.parameters(), lr=lr)
    e2_opt_v = optim.Adam(agent_volatile.e2.parameters(), lr=lr * 3)
    e3_opt_v = optim.Adam(
        list(agent_volatile.e3.parameters()) + list(agent_volatile.latent_stack.parameters()),
        lr=lr,
    )

    stable_ep_rvs    = []
    stable_ep_beta   = []
    volatile_ep_rvs  = []
    volatile_ep_beta = []

    print(f"\n[Phase 1] Training stable ({n_train} eps) and volatile ({n_train} eps)...", flush=True)
    agent_stable.train()
    agent_volatile.train()

    for ep in range(n_train):
        m_s = run_episode(agent_stable,   env_stable,   train=True,
                          e1_opt=e1_opt_s, e2_opt=e2_opt_s, e3_opt=e3_opt_s)
        m_v = run_episode(agent_volatile, env_volatile, train=True,
                          e1_opt=e1_opt_v, e2_opt=e2_opt_v, e3_opt=e3_opt_v)

        stable_ep_rvs.append(m_s["mean_running_variance"])
        stable_ep_beta.append(m_s["mean_z_beta_norm"])
        volatile_ep_rvs.append(m_v["mean_running_variance"])
        volatile_ep_beta.append(m_v["mean_z_beta_norm"])

        if (ep + 1) % 50 == 0:
            print(
                f"  ep {ep+1:3d}: "
                f"stable   rv={m_s['mean_running_variance']:.4f} z_b={m_s['mean_z_beta_norm']:.4f} | "
                f"volatile rv={m_v['mean_running_variance']:.4f} z_b={m_v['mean_z_beta_norm']:.4f}",
                flush=True,
            )

    # ---- Phase 2: Eval -------------------------------------------------------
    print(f"\n[Phase 2] Evaluating ({n_eval} eps each)...", flush=True)
    agent_stable.eval()
    agent_volatile.eval()

    eval_stable_beta   = []
    eval_volatile_beta = []
    eval_stable_rv     = []
    eval_volatile_rv   = []

    for _ep in range(n_eval):
        m_s = run_episode(agent_stable,   env_stable,   train=False)
        m_v = run_episode(agent_volatile, env_volatile, train=False)
        eval_stable_beta.append(m_s["mean_z_beta_norm"])
        eval_volatile_beta.append(m_v["mean_z_beta_norm"])
        eval_stable_rv.append(m_s["mean_running_variance"])
        eval_volatile_rv.append(m_v["mean_running_variance"])

    mean_stable_beta   = float(np.mean(eval_stable_beta))
    mean_volatile_beta = float(np.mean(eval_volatile_beta))
    mean_stable_rv     = float(np.mean(eval_stable_rv))
    mean_volatile_rv   = float(np.mean(eval_volatile_rv))

    c1_delta = mean_volatile_beta - mean_stable_beta
    c1_pass  = c1_delta > C1_DELTA_THRESHOLD

    all_rvs   = stable_ep_rvs + volatile_ep_rvs
    all_betas = stable_ep_beta + volatile_ep_beta
    c2_r    = pearson_r(all_rvs, all_betas)
    c2_pass = c2_r > C2_PEARSON_THRESHOLD

    print(f"\n  Stable  : mean_rv={mean_stable_rv:.4f},  mean_z_beta={mean_stable_beta:.4f}", flush=True)
    print(f"  Volatile: mean_rv={mean_volatile_rv:.4f}, mean_z_beta={mean_volatile_beta:.4f}", flush=True)
    print(f"  C1 delta={c1_delta:+.4f} (threshold {C1_DELTA_THRESHOLD}) -> {'PASS' if c1_pass else 'FAIL'}", flush=True)
    print(f"  C2 Pearson r={c2_r:.4f} (threshold {C2_PEARSON_THRESHOLD}) -> {'PASS' if c2_pass else 'FAIL'}", flush=True)

    # ---- Phase 3: Transfer test ----------------------------------------------
    print(f"\n[Phase 3] Transfer test: stable agent -> volatile ({n_transfer} eps)...", flush=True)
    env_transfer = make_env(volatile=True, seed=seed + 999)

    baseline_beta = float(np.mean(eval_stable_beta[-C3_WINDOW:]))

    transfer_betas = []
    agent_stable.train()
    for ep in range(n_transfer):
        if ep < 100:
            m = run_episode(agent_stable, env_stable, train=False)
        else:
            m = run_episode(agent_stable, env_transfer, train=True,
                            e1_opt=e1_opt_s, e2_opt=e2_opt_s, e3_opt=e3_opt_s)
        transfer_betas.append(m["mean_z_beta_norm"])

    post_switch_beta = float(np.mean(transfer_betas[100:100 + C3_WINDOW]))
    c3_delta = post_switch_beta - baseline_beta
    c3_pass  = c3_delta > C3_DELTA_THRESHOLD

    print(f"  Baseline z_beta (stable): {baseline_beta:.4f}", flush=True)
    print(f"  Post-switch z_beta (ep 100-{100+C3_WINDOW}): {post_switch_beta:.4f}", flush=True)
    print(f"  C3 delta={c3_delta:+.4f} (threshold {C3_DELTA_THRESHOLD}) -> {'PASS' if c3_pass else 'FAIL'}", flush=True)

    # ---- Summary -------------------------------------------------------------
    n_pass = sum([c1_pass, c2_pass, c3_pass])
    overall_pass = n_pass >= 2  # majority vote (2 of 3)
    status = "PASS" if overall_pass else "FAIL"

    print(f"\n{'='*60}", flush=True)
    print(f"RESULT: {n_pass}/3 criteria met -> {status}", flush=True)
    print(f"{'='*60}", flush=True)

    metrics = {
        "mean_stable_z_beta_norm":    mean_stable_beta,
        "mean_volatile_z_beta_norm":  mean_volatile_beta,
        "c1_delta":                   c1_delta,
        "c1_pass":                    1.0 if c1_pass else 0.0,
        "c2_pearson_r":               c2_r,
        "c2_pass":                    1.0 if c2_pass else 0.0,
        "baseline_z_beta":            baseline_beta,
        "post_switch_z_beta":         post_switch_beta,
        "c3_delta":                   c3_delta,
        "c3_pass":                    1.0 if c3_pass else 0.0,
        "criteria_met":               float(n_pass),
        "mean_stable_running_variance":   mean_stable_rv,
        "mean_volatile_running_variance": mean_volatile_rv,
        "rv_condition_difference":        mean_volatile_rv - mean_stable_rv,
        "volatility_signal_dim":          float(VOLATILITY_SIGNAL_DIM),
        "n_train_episodes":               float(n_train),
        "n_eval_episodes":                float(n_eval),
    }

    summary_markdown = (
        f"# V3-EXQ-051c -- Q-007 z_beta Volatility Injection\n\n"
        f"**Status:** {status}\n"
        f"**Claims:** Q-007\n\n"
        f"## Results\n\n"
        f"| Condition | mean_rv | mean_z_beta |\n"
        f"|---|---|---|\n"
        f"| Stable   | {mean_stable_rv:.4f} | {mean_stable_beta:.4f} |\n"
        f"| Volatile | {mean_volatile_rv:.4f} | {mean_volatile_beta:.4f} |\n\n"
        f"## PASS Criteria\n\n"
        f"| Criterion | Result |\n|---|---|\n"
        f"| C1: volatile z_beta > stable + {C1_DELTA_THRESHOLD} (delta={c1_delta:+.4f}) | {'PASS' if c1_pass else 'FAIL'} |\n"
        f"| C2: Pearson r(rv, z_beta) > {C2_PEARSON_THRESHOLD} (r={c2_r:.4f}) | {'PASS' if c2_pass else 'FAIL'} |\n"
        f"| C3: transfer rise > {C3_DELTA_THRESHOLD} (delta={c3_delta:+.4f}) | {'PASS' if c3_pass else 'FAIL'} |\n\n"
        f"Criteria met: {n_pass}/3 -> **{status}**\n"
    )

    evidence_direction = "supports" if overall_pass else ("mixed" if n_pass >= 1 else "weakens")

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
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",       type=int,   default=SEED)
    parser.add_argument("--n-train",    type=int,   default=N_TRAIN_EPISODES)
    parser.add_argument("--n-eval",     type=int,   default=N_EVAL_EPISODES)
    parser.add_argument("--n-transfer", type=int,   default=TRANSFER_EPISODES)
    parser.add_argument("--lr",         type=float, default=1e-3)
    args = parser.parse_args()

    result = run(
        seed=args.seed,
        n_train_episodes=args.n_train,
        n_eval_episodes=args.n_eval,
        n_transfer_episodes=args.n_transfer,
        lr=args.lr,
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
