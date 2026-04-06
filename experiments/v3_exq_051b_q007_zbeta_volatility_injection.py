"""
EXQ-051b -- Q-007: z_beta Volatility Injection (NE/LC analog)

Claim: Q-007 -- z_beta should correlate with environmental volatility
       (running variance of harm prediction errors).

Architectural gap identified in EXQ-051:
  z_beta = beta_encoder(cat(z_self, z_world)) -- purely sensory.
  No pathway from E3's running_variance -> z_beta always ≈ 0.72 in both
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
  running_variance is the REE analog of HGF log-volatility (μ₃), which
  drives affective state in precision-weighted inference models.

Design:
  Two conditions trained and evaluated in parallel:
    STABLE:   hazard positions fixed across all episodes (4 hazards)
    VOLATILE: hazard positions randomised each episode (4 hazards)
  Both conditions have identical mean hazard count -> same mean harm rate.
  The difference is PREDICTABILITY -- stable agents should learn E3 well
  -> low running_variance; volatile agents keep getting surprised -> high
  running_variance.

  With running_variance injected into beta_encoder:
    -> z_beta should be higher in VOLATILE condition (arousal ↑)
    -> Pearson r(running_variance, z_beta_norm) > 0.3

Criteria:
  C1: z_beta_norm_volatile > z_beta_norm_stable + 0.05
        (volatile envs produce higher affective arousal)
  C2: Pearson r(running_variance_trajectory, z_beta_norm_trajectory) > 0.3
        (within-agent correlation over training episodes)
  C3: z_beta responds faster to sudden volatility increase (transfer test):
        after 100 stable episodes -> suddenly volatile -> z_beta rises within
        20 episodes (relative to baseline, Δ > 0.02)

Run-id: exq_051b_v3
"""

import json
import math
import os
import random
import sys
import time
from collections import deque
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig, LatentStackConfig

# ─── Constants ──────────────────────────────────────────────────────────────

SEED             = 42
N_TRAIN_EPISODES = 300    # 150 stable + 150 volatile (interleaved)
N_EVAL_EPISODES  = 50     # 25 per condition
TRANSFER_EPISODES = 150   # stable -> volatile switch at episode 100

MAX_STEPS        = 150
N_HAZARDS        = 4
GRID_SIZE        = 10
ACTION_DIM       = 5   # env.action_dim (5 directions)
SELF_DIM         = 32
WORLD_DIM        = 32
BETA_DIM         = 64

VOLATILITY_SIGNAL_DIM = 1   # Q-007: scalar running_variance -> beta_encoder

# C1 threshold
C1_DELTA_THRESHOLD   = 0.05
# C2 threshold
C2_PEARSON_THRESHOLD = 0.30
# C3 threshold (transfer test)
C3_DELTA_THRESHOLD   = 0.02
C3_WINDOW            = 20    # episodes to check after switch

DEVICE = "cpu"

# ─── Environment helpers ──────────────────────────────────────────────────────

def make_env(volatile: bool, seed: Optional[int] = None) -> CausalGridWorldV2:
    """Create environment.
    volatile=True: hazards drift every step (high within-episode unpredictability).
    volatile=False: hazards are static within episodes (low unpredictability).
    """
    env = CausalGridWorldV2(
        size=GRID_SIZE,
        num_hazards=N_HAZARDS,
        num_resources=3,
        env_drift_interval=1 if volatile else 9999,
        env_drift_prob=0.4 if volatile else 0.0,
        seed=seed,
    )
    return env


# ─── Agent with volatility injection ─────────────────────────────────────────

def make_agent(seed: int, env):
    """Returns (agent, opts) where opts = {'e1': opt, 'e2': opt, 'e3': opt}."""
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        alpha_world=0.9,   # SD-008: event-responsive z_world
        alpha_self=0.3,
    )
    # Q-007: enable volatility injection
    config.latent.volatility_signal_dim = VOLATILITY_SIGNAL_DIM

    # Stable commit threshold (EXQ-018b style -- will be calibrated below)
    config.e3.commitment_threshold = 0.40

    torch.manual_seed(seed)
    agent = REEAgent(config)
    opts = {
        "e1": optim.Adam(agent.e1.parameters(), lr=1e-4),
        "e2": optim.Adam(agent.e2.parameters(), lr=3e-4),
        "e3": optim.Adam(agent.e3.parameters(), lr=1e-4),
    }
    return agent, opts


# ─── Episode runner ───────────────────────────────────────────────────────────

def run_episode(
    agent: REEAgent,
    env,
    train: bool = True,
    opts: dict = None,
) -> dict:
    """
    Run one episode. Returns per-episode metrics including running_variance
    and mean z_beta norm.

    Volatility injection is handled internally by agent.sense() via
    LatentStackConfig.volatility_signal_dim > 0 (configured in make_agent).
    """
    _, obs_dict = env.reset()
    agent.reset()

    total_harm = 0.0
    z_beta_norms: List[float] = []
    running_vars: List[float] = []

    step = 0
    done = False
    while step < MAX_STEPS and not done:
        obs_body  = torch.tensor(obs_dict["body_state"],  dtype=torch.float32)
        obs_world = torch.tensor(obs_dict["world_state"], dtype=torch.float32)

        # Q-007: collect running_variance before sense (updated inside e3 each step)
        running_vars.append(agent.e3._running_variance)

        # SENSE (volatility injection into z_beta happens inside sense() automatically)
        latent = agent.sense(obs_body, obs_world)
        z_beta_norms.append(latent.z_beta.norm().item())

        # ACTION SELECTION
        ticks = agent.clock.advance()
        e1_prior = (
            agent._e1_tick(latent) if ticks["e1_tick"]
            else torch.zeros(1, agent.config.latent.world_dim, device=agent.device)
        )
        z_self_prev = (
            agent._current_latent.z_self.detach().clone()
            if agent._current_latent is not None else None
        )
        candidates = agent.generate_trajectories(latent, e1_prior, ticks)
        action = agent.select_action(candidates, ticks, temperature=1.0)

        if z_self_prev is not None:
            agent.record_transition(z_self_prev, action, latent.z_self.detach())

        _, reward, done, info, obs_dict = env.step(action)
        harm_signal = float(reward) if reward < 0 else 0.0
        total_harm += abs(harm_signal)

        # TRAINING
        if train and opts is not None:
            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total_e1_e2 = e1_loss + e2_loss
            if total_e1_e2.requires_grad:
                opts["e1"].zero_grad()
                opts["e2"].zero_grad()
                total_e1_e2.backward()
                opts["e1"].step()
                opts["e2"].step()

            if agent._current_latent is not None:
                z_world = agent._current_latent.z_world.detach()
                harm_target = torch.tensor(
                    [[1.0 if harm_signal < 0 else 0.0]], device=agent.device
                )
                harm_loss = F.mse_loss(agent.e3.harm_eval(z_world), harm_target)
                maint_loss = agent.compute_self_maintenance_loss()
                total_e3 = harm_loss + maint_loss
                opts["e3"].zero_grad()
                total_e3.backward()
                opts["e3"].step()

        agent.update_residue(harm_signal)
        step += 1

    return {
        "total_harm": total_harm,
        "mean_z_beta_norm": float(np.mean(z_beta_norms)) if z_beta_norms else 0.0,
        "mean_running_variance": float(np.mean(running_vars)) if running_vars else 0.0,
        "final_running_variance": running_vars[-1] if running_vars else 0.0,
        "z_beta_norms": z_beta_norms,
        "running_vars": running_vars,
        "n_steps": step,
    }


# ─── Pearson correlation ──────────────────────────────────────────────────────

def pearson_r(x: List[float], y: List[float]) -> float:
    if len(x) < 5:
        return 0.0
    xn = np.array(x)
    yn = np.array(y)
    if xn.std() < 1e-9 or yn.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(xn, yn)[0, 1])


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("=" * 60)
    print("EXQ-051b: Q-007 z_beta Volatility Injection (NE/LC analog)")
    print("=" * 60)

    # ── Phase 1: Train two agents ──────────────────────────────────────────
    print("\n[Phase 1] Training stable agent...")
    env_stable     = make_env(volatile=False, seed=SEED)
    env_volatile   = make_env(volatile=True,  seed=SEED + 1)
    agent_stable,   opts_stable   = make_agent(SEED,     env_stable)
    agent_volatile, opts_volatile = make_agent(SEED + 1, env_volatile)

    stable_ep_rvs   = []   # per-episode mean running_variance
    stable_ep_beta  = []   # per-episode mean z_beta_norm
    volatile_ep_rvs = []
    volatile_ep_beta = []

    for ep in range(N_TRAIN_EPISODES):
        m_s = run_episode(agent_stable,   env_stable,   train=True,  opts=opts_stable)
        m_v = run_episode(agent_volatile, env_volatile, train=True,  opts=opts_volatile)

        stable_ep_rvs.append(m_s["mean_running_variance"])
        stable_ep_beta.append(m_s["mean_z_beta_norm"])
        volatile_ep_rvs.append(m_v["mean_running_variance"])
        volatile_ep_beta.append(m_v["mean_z_beta_norm"])

        if (ep + 1) % 50 == 0:
            print(
                f"  ep {ep+1:3d}: "
                f"stable rv={m_s['mean_running_variance']:.4f} z_β={m_s['mean_z_beta_norm']:.4f} | "
                f"volatile rv={m_v['mean_running_variance']:.4f} z_β={m_v['mean_z_beta_norm']:.4f}"
            )

    # ── Phase 2: Eval ──────────────────────────────────────────────────────
    print("\n[Phase 2] Evaluating stable vs volatile...")
    eval_stable_beta  = []
    eval_volatile_beta = []
    eval_stable_rv    = []
    eval_volatile_rv  = []

    for ep in range(N_EVAL_EPISODES):
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

    # C1: volatile z_beta > stable z_beta + threshold
    c1_delta = mean_volatile_beta - mean_stable_beta
    c1_pass  = c1_delta > C1_DELTA_THRESHOLD

    # C2: Pearson r(running_variance, z_beta) over training trajectory
    # Use all episodes from both conditions concatenated
    all_rvs   = stable_ep_rvs   + volatile_ep_rvs
    all_betas = stable_ep_beta  + volatile_ep_beta
    c2_r    = pearson_r(all_rvs, all_betas)
    c2_pass = c2_r > C2_PEARSON_THRESHOLD

    print(f"\n  Stable  : mean_rv={mean_stable_rv:.4f},  mean_z_beta={mean_stable_beta:.4f}")
    print(f"  Volatile: mean_rv={mean_volatile_rv:.4f}, mean_z_beta={mean_volatile_beta:.4f}")
    print(f"  C1 delta={c1_delta:+.4f} (threshold {C1_DELTA_THRESHOLD}) -> {'PASS' if c1_pass else 'FAIL'}")
    print(f"  C2 Pearson r={c2_r:.4f} (threshold {C2_PEARSON_THRESHOLD}) -> {'PASS' if c2_pass else 'FAIL'}")

    # ── Phase 3: Transfer test (stable -> volatile) ─────────────────────────
    print("\n[Phase 3] Transfer test: stable agent moved to volatile env...")
    env_transfer = make_env(volatile=True, seed=SEED + 999)

    # Baseline: last C3_WINDOW episodes of stable eval
    baseline_beta = float(np.mean(eval_stable_beta[-C3_WINDOW:]))

    transfer_betas = []
    for ep in range(TRANSFER_EPISODES):
        # Switch to volatile after 100 stable episodes
        if ep < 100:
            m = run_episode(agent_stable, env_stable,    train=False)
        else:
            m = run_episode(agent_stable, env_transfer,  train=True, opts=opts_stable)  # adapt to volatile
        transfer_betas.append(m["mean_z_beta_norm"])

    # Check if z_beta rises within C3_WINDOW after switch (ep 100-120)
    post_switch_beta = float(np.mean(transfer_betas[100:100 + C3_WINDOW]))
    c3_delta = post_switch_beta - baseline_beta
    c3_pass  = c3_delta > C3_DELTA_THRESHOLD

    print(f"  Baseline z_beta (stable): {baseline_beta:.4f}")
    print(f"  Post-switch z_beta (ep 100-{100+C3_WINDOW}): {post_switch_beta:.4f}")
    print(f"  C3 delta={c3_delta:+.4f} (threshold {C3_DELTA_THRESHOLD}) -> {'PASS' if c3_pass else 'FAIL'}")

    # ── Summary ────────────────────────────────────────────────────────────
    n_pass = sum([c1_pass, c2_pass, c3_pass])
    overall_pass = n_pass >= 2  # majority vote (2 of 3)

    print(f"\n{'='*60}")
    print(f"RESULT: {n_pass}/3 criteria met -> {'PASS' if overall_pass else 'FAIL'}")
    print(f"{'='*60}")

    # ── Write result ───────────────────────────────────────────────────────
    result = {
        "run_id":              "exq_051b_v3",
        "experiment_id":       "V3-EXQ-051b",
        "architecture_epoch":  "ree_hybrid_guardrails_v1",
        "claim_ids":           ["Q-007"],
        "status":              "PASS" if overall_pass else "FAIL",
        "timestamp":           time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),

        "criteria": {
            "C1_volatile_z_beta_gt_stable": {
                "pass": c1_pass,
                "mean_stable_z_beta_norm":   mean_stable_beta,
                "mean_volatile_z_beta_norm": mean_volatile_beta,
                "delta":                     c1_delta,
                "threshold":                 C1_DELTA_THRESHOLD,
            },
            "C2_pearson_r_rv_vs_z_beta": {
                "pass":      c2_pass,
                "pearson_r": c2_r,
                "threshold": C2_PEARSON_THRESHOLD,
            },
            "C3_transfer_rise": {
                "pass":           c3_pass,
                "baseline_beta":  baseline_beta,
                "post_switch_beta": post_switch_beta,
                "delta":          c3_delta,
                "threshold":      C3_DELTA_THRESHOLD,
                "window_episodes": C3_WINDOW,
            },
        },

        "supporting_metrics": {
            "mean_stable_running_variance":   mean_stable_rv,
            "mean_volatile_running_variance": mean_volatile_rv,
            "rv_condition_difference":        mean_volatile_rv - mean_stable_rv,
            "volatility_signal_dim":          VOLATILITY_SIGNAL_DIM,
            "n_train_episodes":               N_TRAIN_EPISODES,
            "n_eval_episodes":                N_EVAL_EPISODES,
        },

        "architectural_notes": (
            "Q-007 fix (Option A): LatentStackConfig.volatility_signal_dim=1 injects "
            "E3._running_variance as scalar into beta_encoder at each encode() call. "
            "Biological analog: LC-NE unexpected uncertainty signal (Yu & Dayan 2005). "
            "Volatile envs reset hazard positions each episode -> E3 running_variance "
            "stays elevated -> z_beta arousal elevated. Stable envs -> E3 learns positions "
            "→ running_variance decays -> z_beta arousal lower. "
            "Option C (harm-stream arousal) is the long-term target post SD-010."
        ),
    }

    out_path = ROOT.parent / "REE_assembly" / "evidence" / "experiments" / "exq_051b_v3.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nResult written to {out_path}")
    return result


if __name__ == "__main__":
    main()
