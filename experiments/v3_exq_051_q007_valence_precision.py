"""
V3-EXQ-051 — Q-007: Valence-Precision Correlation (V3 Clean)

Claims: Q-007

Motivation (2026-03-19):
  Q-007 (V2 FAIL): tests whether z_beta (valence/arousal) correlates with E3-derived
  precision (inverse of running_variance). V2 failed because:
    - z_beta was mixed into z_gamma (unified, not separated)
    - E3 precision was hardcoded (not dynamic)

  V3 fixes: z_beta is now isolated in the shared stack (SD-005 split), and E3
  derives precision from its own prediction error variance (ARC-016).

  Prediction:
    Environments with high uncertainty (frequent env drift, varied hazard patterns)
    should produce:
      - Higher z_beta magnitude (arousal elevated by novelty/uncertainty)
      - Higher E3 running_variance (lower precision, more errors)
    → Positive correlation between z_beta norm and running_variance

  Methodology:
    1. Train agent in a stable env (low drift). Record (z_beta_norm, running_variance) pairs.
    2. Train agent in a volatile env (high drift). Record same pairs.
    3. Compute Pearson correlation between z_beta_norm and running_variance within each condition.
    4. Compare mean_z_beta_norm and mean_running_variance across conditions.

PASS criteria (ALL must hold):
  C1: pearson_r_volatile > 0.05         (positive correlation in volatile env)
  C2: mean_z_beta_norm_volatile > mean_z_beta_norm_stable
                                         (arousal higher in volatile env)
  C3: mean_running_var_volatile > mean_running_var_stable
                                         (E3 less certain in volatile env)
  C4: n_pairs_stable >= 50 AND n_pairs_volatile >= 50
  C5: No fatal errors
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
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_051_q007_valence_precision"
CLAIM_IDS = ["Q-007"]


def _action_to_onehot(action_idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, action_idx] = 1.0
    return v


def _mean_safe(lst: List[float]) -> float:
    return float(sum(lst) / len(lst)) if lst else 0.0


def _pearson_r(xs: List[float], ys: List[float]) -> float:
    """Pearson correlation coefficient."""
    if len(xs) < 3:
        return 0.0
    n = len(xs)
    x_t = torch.tensor(xs, dtype=torch.float32)
    y_t = torch.tensor(ys, dtype=torch.float32)
    x_m = x_t - x_t.mean()
    y_m = y_t - y_t.mean()
    cov = (x_m * y_m).sum()
    denom = torch.sqrt((x_m ** 2).sum() * (y_m ** 2).sum())
    if denom < 1e-8:
        return 0.0
    return float((cov / denom).item())


def _train_and_collect(
    seed: int,
    env_drift_interval: int,
    env_drift_prob: float,
    num_episodes: int,
    steps_per_episode: int,
    self_dim: int,
    world_dim: int,
    alpha_world: float,
    label: str,
) -> Dict:
    """
    Train agent in given env and collect (z_beta_norm, running_variance) pairs.
    E3 running_variance is updated each step from z_world prediction error.
    """
    torch.manual_seed(seed)
    random.seed(seed)

    env = CausalGridWorldV2(
        seed=seed, size=12, num_hazards=4, num_resources=5,
        hazard_harm=0.02,
        env_drift_interval=env_drift_interval, env_drift_prob=env_drift_prob,
        proximity_harm_scale=0.05,
        proximity_benefit_scale=0.03,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
    )
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=alpha_world,
        alpha_self=0.3,
        reafference_action_dim=env.action_dim,
    )
    agent = REEAgent(config)

    optimizer = optim.Adam(list(agent.e1.parameters()), lr=1e-3)
    wf_optimizer = optim.Adam(
        list(agent.e2.world_transition.parameters()) +
        list(agent.e2.world_action_encoder.parameters()),
        lr=1e-3,
    )
    wf_buf: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    agent.train()
    z_beta_norms: List[float] = []
    running_vars: List[float] = []
    e3_tick_total = 0

    for ep in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        z_world_prev: Optional[torch.Tensor] = None
        action_prev:  Optional[torch.Tensor] = None
        z_self_prev:  Optional[torch.Tensor] = None

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            latent = agent.sense(obs_body, obs_world)

            if z_self_prev is not None and action_prev is not None:
                agent.record_transition(z_self_prev, action_prev, latent.z_self.detach())

            ticks    = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent) if ticks.get("e1_tick", False)
                else torch.zeros(1, world_dim, device=agent.device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            z_world_curr = latent.z_world.detach()

            if ticks.get("e3_tick", False) and candidates:
                e3_tick_total += 1
                result = agent.e3.select(candidates, temperature=1.0)
                action = result.selected_action.detach()
                agent._last_action = action

                # Update running_variance from E2 world prediction error
                if z_world_prev is not None and action_prev is not None:
                    with torch.no_grad():
                        z_pred = agent.e2.world_forward(z_world_prev, action_prev)
                        pred_error = (z_world_curr - z_pred).detach()
                        agent.e3.update_running_variance(pred_error)

                # Record pair: z_beta norm and current running_variance
                z_beta_norm = float(latent.z_beta.norm(dim=-1).mean().item())
                running_var  = agent.e3._running_variance
                z_beta_norms.append(z_beta_norm)
                running_vars.append(running_var)
            else:
                action = agent._last_action
                if action is None:
                    action = _action_to_onehot(
                        random.randint(0, env.action_dim - 1), env.action_dim, agent.device
                    )
                    agent._last_action = action

            flat_obs, _, done, info, obs_dict = env.step(action)

            if z_world_prev is not None and action_prev is not None:
                wf_buf.append((z_world_prev.cpu(), action_prev.cpu(), z_world_curr.cpu()))
                if len(wf_buf) > 2000:
                    wf_buf = wf_buf[-2000:]

            e1_loss = agent.compute_prediction_loss()
            if e1_loss.requires_grad:
                optimizer.zero_grad()
                e1_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.e1.parameters(), 1.0)
                optimizer.step()

            if len(wf_buf) >= 16:
                k = min(32, len(wf_buf))
                idxs = torch.randperm(len(wf_buf))[:k].tolist()
                zw_b  = torch.cat([wf_buf[i][0] for i in idxs]).to(agent.device)
                a_b   = torch.cat([wf_buf[i][1] for i in idxs]).to(agent.device)
                zw1_b = torch.cat([wf_buf[i][2] for i in idxs]).to(agent.device)
                wf_loss = F.mse_loss(agent.e2.world_forward(zw_b, a_b), zw1_b)
                if wf_loss.requires_grad:
                    wf_optimizer.zero_grad()
                    wf_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(agent.e2.world_transition.parameters()) +
                        list(agent.e2.world_action_encoder.parameters()),
                        1.0,
                    )
                    wf_optimizer.step()

            z_world_prev = z_world_curr
            z_self_prev  = latent.z_self.detach()
            action_prev  = action.detach()
            if done:
                break

        if (ep + 1) % 100 == 0 or ep == num_episodes - 1:
            print(
                f"  [{label}] ep {ep+1}/{num_episodes}"
                f"  z_beta_norm_mean={_mean_safe(z_beta_norms[-100:]):.3f}"
                f"  running_var_mean={_mean_safe(running_vars[-100:]):.4f}",
                flush=True,
            )

    pearson_r = _pearson_r(z_beta_norms, running_vars)
    print(
        f"  [{label}] n_pairs={len(z_beta_norms)}"
        f"  pearson_r={pearson_r:.4f}"
        f"  mean_z_beta={_mean_safe(z_beta_norms):.4f}"
        f"  mean_var={_mean_safe(running_vars):.5f}",
        flush=True,
    )

    return {
        "z_beta_norms":      z_beta_norms,
        "running_vars":      running_vars,
        "pearson_r":         pearson_r,
        "mean_z_beta_norm":  _mean_safe(z_beta_norms),
        "mean_running_var":  _mean_safe(running_vars),
        "n_pairs":           len(z_beta_norms),
        "e3_tick_total":     e3_tick_total,
    }


def run(
    seed: int = 0,
    warmup_episodes: int = 400,
    steps_per_episode: int = 200,
    self_dim: int = 32,
    world_dim: int = 32,
    alpha_world: float = 0.9,
    **kwargs,
) -> dict:

    print(
        f"[V3-EXQ-051] Q-007: Valence-Precision Correlation\n"
        f"  seed={seed}  warmup={warmup_episodes}  alpha_world={alpha_world}",
        flush=True,
    )

    print("\n[V3-EXQ-051] Condition A: STABLE env (low drift)...", flush=True)
    stable_out = _train_and_collect(
        seed=seed,
        env_drift_interval=20, env_drift_prob=0.02,
        num_episodes=warmup_episodes, steps_per_episode=steps_per_episode,
        self_dim=self_dim, world_dim=world_dim,
        alpha_world=alpha_world, label="stable",
    )

    print("\n[V3-EXQ-051] Condition B: VOLATILE env (high drift)...", flush=True)
    volatile_out = _train_and_collect(
        seed=seed,
        env_drift_interval=3, env_drift_prob=0.5,
        num_episodes=warmup_episodes, steps_per_episode=steps_per_episode,
        self_dim=self_dim, world_dim=world_dim,
        alpha_world=alpha_world, label="volatile",
    )

    # PASS / FAIL
    c1_pass = volatile_out["pearson_r"] > 0.05
    c2_pass = volatile_out["mean_z_beta_norm"] > stable_out["mean_z_beta_norm"]
    c3_pass = volatile_out["mean_running_var"] > stable_out["mean_running_var"]
    c4_pass = stable_out["n_pairs"] >= 50 and volatile_out["n_pairs"] >= 50
    c5_pass = True  # no exceptions if we got here

    all_pass    = c1_pass and c2_pass and c3_pass and c4_pass and c5_pass
    status      = "PASS" if all_pass else "FAIL"
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass])

    failure_notes = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: pearson_r_volatile={volatile_out['pearson_r']:.4f} <= 0.05"
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: mean_z_beta_volatile={volatile_out['mean_z_beta_norm']:.4f}"
            f" <= stable={stable_out['mean_z_beta_norm']:.4f}"
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: mean_running_var_volatile={volatile_out['mean_running_var']:.5f}"
            f" <= stable={stable_out['mean_running_var']:.5f}"
        )
    if not c4_pass:
        failure_notes.append(
            f"C4 FAIL: n_pairs stable={stable_out['n_pairs']} volatile={volatile_out['n_pairs']}"
        )

    print(f"\nV3-EXQ-051 verdict: {status}  ({criteria_met}/5)", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)

    metrics = {
        "pearson_r_stable":        float(stable_out["pearson_r"]),
        "pearson_r_volatile":      float(volatile_out["pearson_r"]),
        "mean_z_beta_stable":      float(stable_out["mean_z_beta_norm"]),
        "mean_z_beta_volatile":    float(volatile_out["mean_z_beta_norm"]),
        "mean_running_var_stable":   float(stable_out["mean_running_var"]),
        "mean_running_var_volatile": float(volatile_out["mean_running_var"]),
        "n_pairs_stable":          float(stable_out["n_pairs"]),
        "n_pairs_volatile":        float(volatile_out["n_pairs"]),
        "e3_tick_stable":          float(stable_out["e3_tick_total"]),
        "e3_tick_volatile":        float(volatile_out["e3_tick_total"]),
        "crit1_pass": 1.0 if c1_pass else 0.0,
        "crit2_pass": 1.0 if c2_pass else 0.0,
        "crit3_pass": 1.0 if c3_pass else 0.0,
        "crit4_pass": 1.0 if c4_pass else 0.0,
        "crit5_pass": 1.0 if c5_pass else 0.0,
        "criteria_met": float(criteria_met),
    }

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(f"- {n}" for n in failure_notes)

    summary_markdown = f"""# V3-EXQ-051 — Q-007: Valence-Precision Correlation (V3)

**Status:** {status}
**Claim:** Q-007 — z_beta (valence) correlates with E3-derived precision
**alpha_world:** {alpha_world}  (SD-008)
**Training:** {warmup_episodes} eps per condition (stable drift=0.02, volatile drift=0.5)
**Seed:** {seed}

## Motivation

Q-007 V2 FAIL: z_beta mixed into z_gamma (no SD-005 split) + hardcoded precision.
V3: z_beta is isolated in shared stack; E3 precision derives from prediction error variance.

## Valence-Precision Correlation

| Condition | pearson_r | mean z_beta norm | mean running_var | n_pairs |
|-----------|-----------|-----------------|-----------------|---------|
| Stable (drift=0.02) | {stable_out['pearson_r']:.4f} | {stable_out['mean_z_beta_norm']:.4f} | {stable_out['mean_running_var']:.5f} | {stable_out['n_pairs']} |
| Volatile (drift=0.5) | {volatile_out['pearson_r']:.4f} | {volatile_out['mean_z_beta_norm']:.4f} | {volatile_out['mean_running_var']:.5f} | {volatile_out['n_pairs']} |

## PASS Criteria

| Criterion | Result | Value |
|---|---|---|
| C1: pearson_r_volatile > 0.05 (positive correlation) | {"PASS" if c1_pass else "FAIL"} | {volatile_out['pearson_r']:.4f} |
| C2: z_beta_volatile > z_beta_stable (arousal up) | {"PASS" if c2_pass else "FAIL"} | {volatile_out['mean_z_beta_norm']:.4f} vs {stable_out['mean_z_beta_norm']:.4f} |
| C3: running_var_volatile > running_var_stable (less certain) | {"PASS" if c3_pass else "FAIL"} | {volatile_out['mean_running_var']:.5f} vs {stable_out['mean_running_var']:.5f} |
| C4: n_pairs >= 50 per condition | {"PASS" if c4_pass else "FAIL"} | stable={stable_out['n_pairs']}  volatile={volatile_out['n_pairs']} |
| C5: No fatal errors | {"PASS" if c5_pass else "FAIL"} | 0 |

Criteria met: {criteria_met}/5 → **{status}**
{failure_section}
"""

    return {
        "status": status,
        "metrics": metrics,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": (
            "supports" if all_pass
            else ("mixed" if criteria_met >= 3 else "weakens")
        ),
        "experiment_type": EXPERIMENT_TYPE,
        "fatal_error_count": 0,
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",        type=int,   default=0)
    parser.add_argument("--warmup",      type=int,   default=400)
    parser.add_argument("--steps",       type=int,   default=200)
    parser.add_argument("--alpha-world", type=float, default=0.9)
    args = parser.parse_args()

    result = run(
        seed=args.seed,
        warmup_episodes=args.warmup,
        steps_per_episode=args.steps,
        alpha_world=args.alpha_world,
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
