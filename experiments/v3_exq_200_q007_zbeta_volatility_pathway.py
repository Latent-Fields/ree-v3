#!/opt/local/bin/python3
"""
V3-EXQ-200 -- Q-007: z_beta Volatility Pathway (rv -> z_beta encoder)

Claims: Q-007
Supersedes: V3-EXQ-051c

Q-007 asserts that z_beta encodes affective dimensions including arousal/volatility
state. Prior experiments (EXQ-051, 051b, 051c) attempted manual volatility injection
but had API/config issues. This experiment tests whether enabling the built-in
volatility_signal_dim=1 pathway (running_variance -> z_beta encoder) allows z_beta
to encode environmental volatility state.

Key substrate fix vs EXQ-051c:
  volatility_signal_dim=1 in LatentStackConfig enables the running_variance -> z_beta
  encoder pathway natively. No manual encode_with_volatility bypass needed -- agent.sense()
  handles it when the config flag is set and volatility_signal is passed.

Design:
  Two conditions: STABLE (drift_prob=0.05) vs VOLATILE (drift_prob=0.3)
  Train agent for 500 episodes per condition, then eval for 50 episodes per condition.
  Config: volatility_signal_dim=1 (enables rv->z_beta pathway).
  During eval, record z_beta norm and running_variance at each step.
  Compute Pearson r(running_variance, z_beta_norm) per condition.
  Transfer test: trained stable agent evaluated in volatile env.

  CausalGridWorldV2(size=6, n_hazards=4, nav_bias=0.45)
  nav_bias = 0.45: 45% chance of biased action toward nearest hazard
  (ensures sufficient harm contact for running_variance variation).

  2 seeds: [42, 123]
  Steps per episode: 200

PASS criteria (3/3 required):
  C1: volatile mean_z_beta > stable mean_z_beta + 0.05
      (z_beta encodes volatility state)
  C2: Pearson r(running_variance, z_beta_norm) > 0.3 in at least one condition
      (within-trajectory correlation)
  C3: transfer_rise > 0.02
      (z_beta shifts when switching from stable to volatile mid-eval)

Decision scoring:
  3/3 -> PASS, supports
  2/3 -> FAIL, mixed
  1/3 -> FAIL, mixed
  0/3 -> FAIL, weakens
"""

import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MATCHED_SEEDS    = [42, 123]
N_TRAIN_EPISODES = 500
N_EVAL_EPISODES  = 50
STEPS_PER_EP     = 200
NAV_BIAS         = 0.45

N_HAZARDS    = 4
GRID_SIZE    = 6
ACTION_DIM   = 4
SELF_DIM     = 32
WORLD_DIM    = 32

# Transfer test: train on STABLE for TRANSFER_STABLE_EPS, then switch to volatile
TRANSFER_STABLE_EPS  = 30
TRANSFER_VOLATILE_EPS = 30
TRANSFER_WINDOW       = 20

C1_DELTA_THRESHOLD   = 0.05
C2_PEARSON_THRESHOLD = 0.30
C3_DELTA_THRESHOLD   = 0.02

EXPERIMENT_TYPE = "v3_exq_200_q007_zbeta_volatility_pathway"
CLAIM_IDS = ["Q-007"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _action_to_onehot(idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
    return v


def _hazard_approach_action(env, n_actions: int) -> int:
    """Return action index that moves agent toward nearest hazard."""
    ax, ay = env.agent_x, env.agent_y
    best_dist = float("inf")
    best_action = random.randint(0, n_actions - 1)
    for hx, hy in env.hazards:
        dist = abs(hx - ax) + abs(hy - ay)
        if dist < best_dist:
            best_dist = dist
            dr = hx - ax
            dc = hy - ay
            if abs(dr) >= abs(dc):
                best_action = 0 if dr < 0 else 1  # up/down
            else:
                best_action = 2 if dc < 0 else 3  # left/right
    return best_action


def pearson_r(x: List[float], y: List[float]) -> float:
    if len(x) < 5:
        return 0.0
    xn = np.array(x)
    yn = np.array(y)
    if xn.std() < 1e-9 or yn.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(xn, yn)[0, 1])


def _mean_safe(lst: List[float]) -> float:
    return float(sum(lst) / len(lst)) if lst else 0.0


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_env(drift_prob: float, seed: Optional[int] = None) -> CausalGridWorldV2:
    """Create CausalGridWorldV2 with specified drift_prob."""
    return CausalGridWorldV2(
        size=GRID_SIZE,
        num_hazards=N_HAZARDS,
        num_resources=3,
        seed=seed,
        env_drift_prob=drift_prob,
        env_drift_interval=1 if drift_prob > 0.1 else 50,
        hazard_field_decay=0.5,
        resource_field_decay=0.5,
        proximity_harm_scale=0.05,
        proximity_benefit_scale=0.03,
    )


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def make_agent(seed: int, env) -> REEAgent:
    """Create REEAgent with volatility_signal_dim=1 (Q-007 pathway)."""
    torch.manual_seed(seed)
    random.seed(seed)
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        alpha_world=0.9,       # SD-008
        alpha_self=0.3,
        use_event_classifier=True,  # SD-009
        reafference_action_dim=env.action_dim,
    )
    # Q-007: enable rv -> z_beta pathway
    config.latent.volatility_signal_dim = 1
    config.e3.commitment_threshold = 0.40
    return REEAgent(config)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(
    agent: REEAgent,
    env,
    train: bool = True,
    e1_opt=None,
    e2_opt=None,
    e3_opt=None,
    nav_bias: float = 0.0,
) -> Dict:
    """Run one episode. Returns per-step z_beta_norm and running_variance."""
    flat_obs, obs_dict = env.reset()
    agent.reset()

    z_beta_norms: List[float] = []
    running_vars: List[float] = []
    volatility_ests: List[float] = []
    total_harm = 0.0
    z_world_prev: Optional[torch.Tensor] = None
    action_prev: Optional[torch.Tensor] = None

    for _step in range(STEPS_PER_EP):
        obs_body = obs_dict["body_state"]
        obs_world = obs_dict["world_state"]

        # Record signals before sense() -- sense() reads volatility_estimate
        # when volatility_signal_dim > 0 (Q-007 pathway).
        rv = float(agent.e3._running_variance)
        vol_est = float(agent.e3.volatility_estimate)

        if train:
            latent = agent.sense(obs_body, obs_world)
        else:
            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)

        z_beta_norms.append(float(latent.z_beta.norm().item()))
        running_vars.append(rv)
        volatility_ests.append(vol_est)

        ticks = agent.clock.advance()
        e1_prior = (
            agent._e1_tick(latent) if ticks.get("e1_tick", False)
            else torch.zeros(1, WORLD_DIM, device=agent.device)
        )

        if train:
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            # Nav-biased action selection
            if nav_bias > 0 and random.random() < nav_bias:
                a_idx = _hazard_approach_action(env, env.action_dim)
                action = _action_to_onehot(a_idx, env.action_dim, agent.device)
                agent._last_action = action
            else:
                action = agent.select_action(candidates, ticks, temperature=1.0)
                if action is None:
                    a_idx = random.randint(0, env.action_dim - 1)
                    action = _action_to_onehot(a_idx, env.action_dim, agent.device)
                    agent._last_action = action
        else:
            with torch.no_grad():
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                action = agent.select_action(candidates, ticks, temperature=0.5)
                if action is None:
                    a_idx = random.randint(0, env.action_dim - 1)
                    action = _action_to_onehot(a_idx, env.action_dim, agent.device)
                    agent._last_action = action

        # Training losses
        z_world_curr = latent.z_world.detach()
        if train and e1_opt is not None:
            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            if e1_loss.requires_grad or e2_loss.requires_grad:
                total_loss = e1_loss + e2_loss
                e1_opt.zero_grad()
                e2_opt.zero_grad()
                total_loss.backward()
                e1_opt.step()
                e2_opt.step()

        flat_obs, harm_signal, done, info, obs_dict = env.step(action)
        harm_val = float(harm_signal) if harm_signal < 0 else 0.0
        total_harm += abs(harm_val)

        # E3 harm supervision
        if train and e3_opt is not None and agent._current_latent is not None:
            z_world = agent._current_latent.z_world.detach()
            harm_target = torch.tensor(
                [[1.0 if harm_val < 0 else 0.0]], device=agent.device
            )
            harm_loss = F.mse_loss(agent.e3.harm_eval(z_world), harm_target)
            e3_opt.zero_grad()
            harm_loss.backward()
            e3_opt.step()

        agent.update_residue(harm_val)
        z_world_prev = z_world_curr
        action_prev = action

        if done:
            break

    return {
        "total_harm": total_harm,
        "z_beta_norms": z_beta_norms,
        "running_vars": running_vars,
        "volatility_ests": volatility_ests,
        "mean_z_beta_norm": _mean_safe(z_beta_norms),
        "mean_running_variance": _mean_safe(running_vars),
        "mean_volatility_est": _mean_safe(volatility_ests),
    }


# ---------------------------------------------------------------------------
# Per-seed experiment
# ---------------------------------------------------------------------------

def run_seed(seed: int, dry_run: bool = False) -> Dict:
    """Run full experiment for one seed. Returns per-seed metrics."""
    n_train = 5 if dry_run else N_TRAIN_EPISODES
    n_eval = 3 if dry_run else N_EVAL_EPISODES

    print(f"\n--- Seed {seed} ---", flush=True)

    # Create envs and agents for both conditions
    env_stable = make_env(drift_prob=0.05, seed=seed)
    env_volatile = make_env(drift_prob=0.3, seed=seed + 1000)
    agent_stable = make_agent(seed, env_stable)
    agent_volatile = make_agent(seed + 1, env_volatile)

    lr = 1e-3
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

    # Phase 1: Train
    print(f"  [Phase 1] Training ({n_train} eps each)...", flush=True)
    agent_stable.train()
    agent_volatile.train()

    train_stable_rvs = []
    train_stable_betas = []
    train_volatile_rvs = []
    train_volatile_betas = []

    for ep in range(n_train):
        m_s = run_episode(agent_stable, env_stable, train=True,
                          e1_opt=e1_opt_s, e2_opt=e2_opt_s, e3_opt=e3_opt_s,
                          nav_bias=NAV_BIAS)
        m_v = run_episode(agent_volatile, env_volatile, train=True,
                          e1_opt=e1_opt_v, e2_opt=e2_opt_v, e3_opt=e3_opt_v,
                          nav_bias=NAV_BIAS)

        train_stable_rvs.append(m_s["mean_running_variance"])
        train_stable_betas.append(m_s["mean_z_beta_norm"])
        train_volatile_rvs.append(m_v["mean_running_variance"])
        train_volatile_betas.append(m_v["mean_z_beta_norm"])

        if (ep + 1) % 100 == 0 or (dry_run and ep == n_train - 1):
            print(
                f"    ep {ep+1:3d}: "
                f"stable rv={m_s['mean_running_variance']:.4f} vol={m_s['mean_volatility_est']:.2e} z_b={m_s['mean_z_beta_norm']:.4f} | "
                f"volatile rv={m_v['mean_running_variance']:.4f} vol={m_v['mean_volatility_est']:.2e} z_b={m_v['mean_z_beta_norm']:.4f}",
                flush=True,
            )

    # Phase 2: Eval
    print(f"  [Phase 2] Evaluating ({n_eval} eps each)...", flush=True)
    agent_stable.eval()
    agent_volatile.eval()

    eval_stable_betas = []
    eval_volatile_betas = []
    eval_stable_rvs = []
    eval_volatile_rvs = []
    eval_stable_vols = []
    eval_volatile_vols = []
    # Collect step-level data for Pearson correlation (C2 uses volatility_est,
    # which is the actual signal injected into z_beta)
    eval_stable_step_vol = []
    eval_stable_step_beta = []
    eval_volatile_step_vol = []
    eval_volatile_step_beta = []

    for _ep in range(n_eval):
        m_s = run_episode(agent_stable, env_stable, train=False)
        m_v = run_episode(agent_volatile, env_volatile, train=False)
        eval_stable_betas.append(m_s["mean_z_beta_norm"])
        eval_volatile_betas.append(m_v["mean_z_beta_norm"])
        eval_stable_rvs.append(m_s["mean_running_variance"])
        eval_volatile_rvs.append(m_v["mean_running_variance"])
        eval_stable_vols.append(m_s["mean_volatility_est"])
        eval_volatile_vols.append(m_v["mean_volatility_est"])
        eval_stable_step_vol.extend(m_s["volatility_ests"])
        eval_stable_step_beta.extend(m_s["z_beta_norms"])
        eval_volatile_step_vol.extend(m_v["volatility_ests"])
        eval_volatile_step_beta.extend(m_v["z_beta_norms"])

    mean_stable_beta = _mean_safe(eval_stable_betas)
    mean_volatile_beta = _mean_safe(eval_volatile_betas)
    mean_stable_rv = _mean_safe(eval_stable_rvs)
    mean_volatile_rv = _mean_safe(eval_volatile_rvs)

    # C1: volatile mean_z_beta > stable mean_z_beta + 0.05
    c1_delta = mean_volatile_beta - mean_stable_beta
    c1_pass = c1_delta > C1_DELTA_THRESHOLD

    # C2: Pearson r(volatility_est, z_beta) > 0.3 in at least one condition
    # Uses volatility_estimate (var(rv)) since that is the signal injected
    # into z_beta, not raw rv.
    c2_r_stable = pearson_r(eval_stable_step_vol, eval_stable_step_beta)
    c2_r_volatile = pearson_r(eval_volatile_step_vol, eval_volatile_step_beta)
    c2_r_best = max(c2_r_stable, c2_r_volatile)
    c2_pass = c2_r_best > C2_PEARSON_THRESHOLD

    mean_stable_vol = _mean_safe(eval_stable_vols)
    mean_volatile_vol = _mean_safe(eval_volatile_vols)

    print(f"    Stable:   mean_rv={mean_stable_rv:.4f}, vol_est={mean_stable_vol:.2e}, mean_z_beta={mean_stable_beta:.4f}", flush=True)
    print(f"    Volatile: mean_rv={mean_volatile_rv:.4f}, vol_est={mean_volatile_vol:.2e}, mean_z_beta={mean_volatile_beta:.4f}", flush=True)
    print(f"    C1 delta={c1_delta:+.4f} (threshold {C1_DELTA_THRESHOLD}) -> {'PASS' if c1_pass else 'FAIL'}", flush=True)
    print(f"    C2 r_stable={c2_r_stable:.4f}, r_volatile={c2_r_volatile:.4f}, best={c2_r_best:.4f} "
          f"(threshold {C2_PEARSON_THRESHOLD}) -> {'PASS' if c2_pass else 'FAIL'}", flush=True)

    # Phase 3: Transfer test -- stable agent -> volatile env
    print(f"  [Phase 3] Transfer test...", flush=True)
    env_transfer = make_env(drift_prob=0.3, seed=seed + 9999)

    # Baseline: stable agent in stable env
    agent_stable.eval()
    baseline_betas = []
    for _ep in range(TRANSFER_STABLE_EPS if not dry_run else 3):
        m = run_episode(agent_stable, env_stable, train=False)
        baseline_betas.append(m["mean_z_beta_norm"])
    baseline_beta = _mean_safe(baseline_betas[-TRANSFER_WINDOW:])

    # Switch: stable agent in volatile env (no further training)
    transfer_betas = []
    for _ep in range(TRANSFER_VOLATILE_EPS if not dry_run else 3):
        m = run_episode(agent_stable, env_transfer, train=False)
        transfer_betas.append(m["mean_z_beta_norm"])
    post_switch_beta = _mean_safe(transfer_betas[:TRANSFER_WINDOW])

    c3_delta = post_switch_beta - baseline_beta
    c3_pass = c3_delta > C3_DELTA_THRESHOLD

    print(f"    Baseline z_beta (stable env): {baseline_beta:.4f}", flush=True)
    print(f"    Post-switch z_beta (volatile env): {post_switch_beta:.4f}", flush=True)
    print(f"    C3 delta={c3_delta:+.4f} (threshold {C3_DELTA_THRESHOLD}) -> {'PASS' if c3_pass else 'FAIL'}", flush=True)

    return {
        "seed": seed,
        "mean_stable_z_beta": mean_stable_beta,
        "mean_volatile_z_beta": mean_volatile_beta,
        "mean_stable_rv": mean_stable_rv,
        "mean_volatile_rv": mean_volatile_rv,
        "mean_stable_vol": mean_stable_vol,
        "mean_volatile_vol": mean_volatile_vol,
        "c1_delta": c1_delta,
        "c1_pass": c1_pass,
        "c2_r_stable": c2_r_stable,
        "c2_r_volatile": c2_r_volatile,
        "c2_r_best": c2_r_best,
        "c2_pass": c2_pass,
        "baseline_beta": baseline_beta,
        "post_switch_beta": post_switch_beta,
        "c3_delta": c3_delta,
        "c3_pass": c3_pass,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(**kwargs) -> dict:
    dry_run = kwargs.get("dry_run", False)
    seeds = kwargs.get("seeds", MATCHED_SEEDS)

    print("=" * 60, flush=True)
    print("EXQ-200: Q-007 z_beta Volatility Pathway (rv -> z_beta)", flush=True)
    print("  volatility_signal_dim=1, alpha_world=0.9, use_event_classifier=True", flush=True)
    print(f"  seeds={seeds}, train={N_TRAIN_EPISODES}, eval={N_EVAL_EPISODES}", flush=True)
    print(f"  grid={GRID_SIZE}x{GRID_SIZE}, hazards={N_HAZARDS}, nav_bias={NAV_BIAS}", flush=True)
    print("=" * 60, flush=True)

    t0 = time.time()
    seed_results = []
    for seed in seeds:
        seed_results.append(run_seed(seed, dry_run=dry_run))

    elapsed = time.time() - t0
    print(f"\nTotal wall time: {elapsed:.1f}s", flush=True)

    # Aggregate across seeds
    all_c1 = [r["c1_pass"] for r in seed_results]
    all_c2 = [r["c2_pass"] for r in seed_results]
    all_c3 = [r["c3_pass"] for r in seed_results]

    # Majority vote per criterion across seeds
    c1_majority = sum(all_c1) > len(seeds) / 2
    c2_majority = sum(all_c2) > len(seeds) / 2
    c3_majority = sum(all_c3) > len(seeds) / 2

    n_criteria_pass = sum([c1_majority, c2_majority, c3_majority])
    overall_pass = n_criteria_pass == 3  # need 3/3
    status = "PASS" if overall_pass else "FAIL"

    # Aggregate means
    agg_stable_beta = _mean_safe([r["mean_stable_z_beta"] for r in seed_results])
    agg_volatile_beta = _mean_safe([r["mean_volatile_z_beta"] for r in seed_results])
    agg_stable_rv = _mean_safe([r["mean_stable_rv"] for r in seed_results])
    agg_volatile_rv = _mean_safe([r["mean_volatile_rv"] for r in seed_results])
    agg_stable_vol = _mean_safe([r["mean_stable_vol"] for r in seed_results])
    agg_volatile_vol = _mean_safe([r["mean_volatile_vol"] for r in seed_results])
    agg_c1_delta = _mean_safe([r["c1_delta"] for r in seed_results])
    agg_c2_best = _mean_safe([r["c2_r_best"] for r in seed_results])
    agg_c3_delta = _mean_safe([r["c3_delta"] for r in seed_results])

    print(f"\n{'='*60}", flush=True)
    print(f"AGGREGATE RESULTS ({len(seeds)} seeds)", flush=True)
    print(f"  C1 (volatile > stable + {C1_DELTA_THRESHOLD}): "
          f"{'PASS' if c1_majority else 'FAIL'} ({sum(all_c1)}/{len(seeds)} seeds)", flush=True)
    print(f"  C2 (Pearson r > {C2_PEARSON_THRESHOLD}): "
          f"{'PASS' if c2_majority else 'FAIL'} ({sum(all_c2)}/{len(seeds)} seeds)", flush=True)
    print(f"  C3 (transfer rise > {C3_DELTA_THRESHOLD}): "
          f"{'PASS' if c3_majority else 'FAIL'} ({sum(all_c3)}/{len(seeds)} seeds)", flush=True)
    print(f"\n  Criteria met: {n_criteria_pass}/3 -> {status}", flush=True)
    print(f"{'='*60}", flush=True)

    metrics = {
        "agg_stable_z_beta": agg_stable_beta,
        "agg_volatile_z_beta": agg_volatile_beta,
        "agg_stable_rv": agg_stable_rv,
        "agg_volatile_rv": agg_volatile_rv,
        "agg_stable_vol": agg_stable_vol,
        "agg_volatile_vol": agg_volatile_vol,
        "agg_c1_delta": agg_c1_delta,
        "c1_majority_pass": 1.0 if c1_majority else 0.0,
        "agg_c2_r_best": agg_c2_best,
        "c2_majority_pass": 1.0 if c2_majority else 0.0,
        "agg_c3_delta": agg_c3_delta,
        "c3_majority_pass": 1.0 if c3_majority else 0.0,
        "criteria_met": float(n_criteria_pass),
        "n_seeds": float(len(seeds)),
        "volatility_signal_dim": 1.0,
        "alpha_world": 0.9,
        "use_event_classifier": 1.0,
        "nav_bias": float(NAV_BIAS),
        "n_train_episodes": float(N_TRAIN_EPISODES),
        "n_eval_episodes": float(N_EVAL_EPISODES),
        "steps_per_episode": float(STEPS_PER_EP),
        "elapsed_seconds": elapsed,
    }

    # Per-seed detail
    for i, r in enumerate(seed_results):
        pfx = f"seed_{r['seed']}"
        metrics[f"{pfx}_c1_delta"] = r["c1_delta"]
        metrics[f"{pfx}_c1_pass"] = 1.0 if r["c1_pass"] else 0.0
        metrics[f"{pfx}_c2_r_stable"] = r["c2_r_stable"]
        metrics[f"{pfx}_c2_r_volatile"] = r["c2_r_volatile"]
        metrics[f"{pfx}_c2_r_best"] = r["c2_r_best"]
        metrics[f"{pfx}_c2_pass"] = 1.0 if r["c2_pass"] else 0.0
        metrics[f"{pfx}_c3_delta"] = r["c3_delta"]
        metrics[f"{pfx}_c3_pass"] = 1.0 if r["c3_pass"] else 0.0

    summary_markdown = (
        f"# V3-EXQ-200 -- Q-007 z_beta Volatility Pathway\n\n"
        f"**Status:** {status}\n"
        f"**Claims:** Q-007\n"
        f"**Supersedes:** V3-EXQ-051c\n\n"
        f"## Key Config\n\n"
        f"- volatility_signal_dim=1 (rv -> z_beta encoder pathway)\n"
        f"- alpha_world=0.9 (SD-008)\n"
        f"- use_event_classifier=True (SD-009)\n"
        f"- CausalGridWorldV2 size={GRID_SIZE}, hazards={N_HAZARDS}, nav_bias={NAV_BIAS}\n"
        f"- Seeds: {seeds}\n\n"
        f"## Aggregate Results ({len(seeds)} seeds)\n\n"
        f"| Condition | mean_rv | mean_z_beta |\n"
        f"|---|---|---|\n"
        f"| Stable (drift=0.05) | {agg_stable_rv:.4f} | {agg_stable_beta:.4f} |\n"
        f"| Volatile (drift=0.3) | {agg_volatile_rv:.4f} | {agg_volatile_beta:.4f} |\n\n"
        f"## PASS Criteria\n\n"
        f"| Criterion | Result | Seeds passing |\n|---|---|---|\n"
        f"| C1: volatile z_beta > stable + {C1_DELTA_THRESHOLD} (delta={agg_c1_delta:+.4f}) | "
        f"{'PASS' if c1_majority else 'FAIL'} | {sum(all_c1)}/{len(seeds)} |\n"
        f"| C2: Pearson r(rv, z_beta) > {C2_PEARSON_THRESHOLD} (r={agg_c2_best:.4f}) | "
        f"{'PASS' if c2_majority else 'FAIL'} | {sum(all_c2)}/{len(seeds)} |\n"
        f"| C3: transfer rise > {C3_DELTA_THRESHOLD} (delta={agg_c3_delta:+.4f}) | "
        f"{'PASS' if c3_majority else 'FAIL'} | {sum(all_c3)}/{len(seeds)} |\n\n"
        f"Criteria met: {n_criteria_pass}/3 -> **{status}**\n"
    )

    evidence_direction = (
        "supports" if overall_pass
        else ("mixed" if n_criteria_pass >= 1 else "weakens")
    )

    return {
        "status": status,
        "metrics": metrics,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": evidence_direction,
        "experiment_type": EXPERIMENT_TYPE,
        "supersedes": "v3_exq_051c_q007_zbeta_volatility_injection",
        "fatal_error_count": 0,
    }


if __name__ == "__main__":
    import argparse
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Short run for smoke test (5 train, 3 eval)")
    args = parser.parse_args()

    result = run(dry_run=args.dry_run)

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
