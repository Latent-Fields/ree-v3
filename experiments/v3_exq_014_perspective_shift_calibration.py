"""
V3-EXQ-014 — Perspective Shift Calibration

Claim: SD-007 (encoder.perspective_corrected_world_latent), MECH-098 (encoder
reafference cancellation).

Motivation (2026-03-17):
  EXQ-012 revealed calibration_gap = 0.0007 after switching to signed regression.
  Root cause: E2_world takes an identity shortcut because the egocentric world_obs
  changes on every body movement (perspective shift), not genuine world change.
  E2 correctly learns that z_world barely changes in content, because perspective
  shift dominates and world content is stable.

  This experiment QUANTIFIES that problem: how much of z_world change variance
  is explained by locomotion alone (self-motion + action)?

  Method:
  1. Run 300 episodes collecting (z_self_prev, a_prev, Δz_world, transition_type)
  2. Fit a linear predictor on empty-space steps:
       Δz_world_hat = W @ concat([z_self_prev; a_prev]) + b
  3. Compute R² on held-out empty steps → locomotion_explained_variance
  4. Compute mean ||Δz_world|| per event type → perspective_shift_dominance_ratio

PASS criteria:
  C1: locomotion_explained_variance > 0.3
      Locomotion explains > 30% of z_world variance (confirms the problem is real)
  C2: mean_dz_world_env_hazard > mean_dz_world_empty
      Genuine world events produce larger z_world change than empty locomotion steps
  C3: n_empty >= 200 (enough data for regression)
  C4: No fatal errors
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from typing import Dict, List, Tuple
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorld
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_014_perspective_shift_calibration"
CLAIM_IDS = ["SD-007", "MECH-098"]

E2_ROLLOUT_STEPS = 5
RECON_WEIGHT = 1.0


def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _make_world_decoder(world_dim: int, world_obs_dim: int, hidden_dim: int = 64):
    return nn.Sequential(
        nn.Linear(world_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, world_obs_dim),
    )


def _train_episodes(
    agent: REEAgent,
    env: CausalGridWorld,
    world_decoder: nn.Module,
    optimizer: optim.Optimizer,
    num_episodes: int,
    steps_per_episode: int,
) -> None:
    """Training phase: E1 + E2_self + E2_world + reconstruction."""
    agent.train()
    world_decoder.train()
    traj_buffer: List = []
    MAX_TRAJ_BUFFER = 200

    for ep in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        episode_traj: List = []

        for step in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]

            latent = agent.sense(obs_body, obs_world)
            agent.clock.advance()

            action_idx = random.randint(0, env.action_dim - 1)
            action = _action_to_onehot(action_idx, env.action_dim, agent.device)
            agent._last_action = action

            episode_traj.append((latent.z_world.detach(), action.detach()))

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)

            e1_loss = agent.compute_prediction_loss()
            e2_self_loss = agent.compute_e2_loss()

            # Multi-step E2 world loss
            e2w_loss = _compute_e2w_loss(agent, traj_buffer, E2_ROLLOUT_STEPS)

            obs_w = obs_world.unsqueeze(0) if obs_world.dim() == 1 else obs_world
            z_w = agent.latent_stack.split_encoder.world_encoder(obs_w)
            recon = world_decoder(z_w)
            recon_loss = F.mse_loss(recon, obs_w)

            total = e1_loss + e2_self_loss + e2w_loss + RECON_WEIGHT * recon_loss
            if total.requires_grad:
                optimizer.zero_grad()
                total.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(world_decoder.parameters(), 1.0)
                optimizer.step()

            if done:
                break

        for start in range(0, len(episode_traj) - E2_ROLLOUT_STEPS):
            traj_buffer.append(episode_traj[start:start + E2_ROLLOUT_STEPS + 1])
        if len(traj_buffer) > MAX_TRAJ_BUFFER:
            traj_buffer = traj_buffer[-MAX_TRAJ_BUFFER:]

        if (ep + 1) % 50 == 0 or ep == num_episodes - 1:
            print(f"  [train] ep {ep+1}/{num_episodes}", flush=True)


def _compute_e2w_loss(agent, traj_buffer, rollout_steps, batch_size=8):
    if len(traj_buffer) < 2:
        return next(agent.e1.parameters()).sum() * 0.0
    n = min(batch_size, len(traj_buffer))
    idxs = torch.randperm(len(traj_buffer))[:n].tolist()
    total = next(agent.e1.parameters()).sum() * 0.0
    count = 0
    for idx in idxs:
        seg = traj_buffer[idx]
        if len(seg) < rollout_steps + 1:
            continue
        z = seg[0][0]
        z_target = seg[rollout_steps][0]
        for k in range(rollout_steps):
            z = agent.e2.world_forward(z, seg[k][1])
        total = total + F.mse_loss(z, z_target)
        count += 1
    return total / count if count > 0 else total


def _collect_data(
    agent: REEAgent,
    env: CausalGridWorld,
    num_episodes: int,
    steps_per_episode: int,
) -> Dict[str, List]:
    """Collect (z_self_prev, a_prev, dz_world, transition_type) for all steps."""
    agent.eval()

    data: Dict[str, List] = {
        "z_self_prev": [],   # List of [1, self_dim]
        "a_prev": [],        # List of [1, action_dim]
        "dz_world": [],      # List of float (L2 norm of z_world delta)
        "dz_world_vec": [],  # List of [1, world_dim] (actual delta vector, for linear fit)
        "transition_type": [],
    }

    for episode in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()

        z_world_prev = None
        z_self_prev = None
        a_prev = None

        for step in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]

            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
                agent.clock.advance()

                z_self_curr  = latent.z_self.detach()    # [1, self_dim]
                z_world_curr = latent.z_world.detach()   # [1, world_dim]

                action_idx = random.randint(0, env.action_dim - 1)
                action = _action_to_onehot(action_idx, env.action_dim, agent.device)
                agent._last_action = action

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)
            ttype = info["transition_type"]

            # Record previous step's data once we have delta
            if z_world_prev is not None:
                dz_vec = (z_world_curr - z_world_prev)  # [1, world_dim]
                dz_norm = float(torch.norm(dz_vec).item())

                data["z_self_prev"].append(z_self_prev.cpu())
                data["a_prev"].append(a_prev.cpu())
                data["dz_world"].append(dz_norm)
                data["dz_world_vec"].append(dz_vec.cpu())
                data["transition_type"].append(ttype)

            z_world_prev = z_world_curr
            z_self_prev  = z_self_curr
            a_prev       = action.detach()

            if done:
                break

    return data


def _compute_r_squared(
    X: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    test_mask: torch.Tensor,
) -> Tuple[float, float]:
    """
    Fit linear regression on train_mask, evaluate R² on test_mask.
    X: [N, input_dim], y: [N, output_dim]
    Returns (r2_train, r2_test)
    """
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test  = X[test_mask]
    y_test  = y[test_mask]

    if X_train.shape[0] < 10 or X_test.shape[0] < 5:
        return 0.0, 0.0

    # Add bias column
    ones_train = torch.ones(X_train.shape[0], 1)
    ones_test  = torch.ones(X_test.shape[0], 1)
    X_train_b = torch.cat([X_train, ones_train], dim=1)
    X_test_b  = torch.cat([X_test,  ones_test],  dim=1)

    # Closed-form least squares: W = (X'X)^{-1} X'y
    try:
        W = torch.linalg.lstsq(X_train_b, y_train).solution  # [input_dim+1, output_dim]
    except Exception:
        return 0.0, 0.0

    y_pred_test = X_test_b @ W
    ss_res = ((y_test - y_pred_test) ** 2).sum()
    ss_tot = ((y_test - y_test.mean(dim=0, keepdim=True)) ** 2).sum()
    r2_test = 1.0 - float((ss_res / (ss_tot + 1e-8)).item())

    y_pred_train = X_train_b @ W
    ss_res_train = ((y_train - y_pred_train) ** 2).sum()
    ss_tot_train = ((y_train - y_train.mean(dim=0, keepdim=True)) ** 2).sum()
    r2_train = 1.0 - float((ss_res_train / (ss_tot_train + 1e-8)).item())

    return r2_train, r2_test


def run(
    seed: int = 0,
    num_train_episodes: int = 300,
    num_eval_episodes: int = 100,
    steps_per_episode: int = 200,
    self_dim: int = 32,
    world_dim: int = 32,
    lr: float = 1e-3,
    **kwargs,
) -> dict:
    torch.manual_seed(seed)
    random.seed(seed)

    env = CausalGridWorld(
        seed=seed,
        size=12,
        num_hazards=15,
        num_resources=5,
        env_drift_interval=3,
        env_drift_prob=0.5,
    )
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
    )
    agent = REEAgent(config)

    world_decoder = _make_world_decoder(world_dim, env.world_obs_dim)
    params = list(agent.parameters()) + list(world_decoder.parameters())
    optimizer = optim.Adam(params, lr=lr)

    print(f"[V3-EXQ-014] Training {num_train_episodes} eps...", flush=True)
    _train_episodes(agent, env, world_decoder, optimizer, num_train_episodes, steps_per_episode)

    print(f"[V3-EXQ-014] Collecting {num_eval_episodes} eval eps...", flush=True)
    data = _collect_data(agent, env, num_eval_episodes, steps_per_episode)

    if not data["transition_type"]:
        return {
            "status": "FAIL",
            "metrics": {"fatal_error_count": 1.0},
            "summary_markdown": "# FAIL\n\nNo data collected.",
            "claim_ids": CLAIM_IDS,
            "evidence_direction": "unknown",
            "experiment_type": EXPERIMENT_TYPE,
            "fatal_error_count": 1,
        }

    # ── Build arrays ───────────────────────────────────────────────────
    ttypes     = data["transition_type"]
    dz_norms   = data["dz_world"]
    z_self_all = torch.cat(data["z_self_prev"], dim=0)   # [N, self_dim]
    a_all      = torch.cat(data["a_prev"],      dim=0)   # [N, action_dim]
    dz_vec_all = torch.cat(data["dz_world_vec"],dim=0)   # [N, world_dim]

    empty_mask  = torch.tensor([t == "none"              for t in ttypes], dtype=torch.bool)
    env_mask    = torch.tensor([t == "env_caused_hazard" for t in ttypes], dtype=torch.bool)
    agent_mask  = torch.tensor([t == "agent_caused_hazard" for t in ttypes], dtype=torch.bool)

    n_empty = int(empty_mask.sum().item())
    n_env   = int(env_mask.sum().item())
    n_agent = int(agent_mask.sum().item())

    print(f"  Data counts — empty: {n_empty}  env_hazard: {n_env}  agent_hazard: {n_agent}",
          flush=True)

    # ── Mean ||Δz_world|| by event type ───────────────────────────────
    def _mean_norm(mask):
        vals = [dz_norms[i] for i, m in enumerate(mask.tolist()) if m]
        return float(sum(vals) / max(1, len(vals))) if vals else 0.0

    mean_dz_empty  = _mean_norm(empty_mask)
    mean_dz_env    = _mean_norm(env_mask)
    mean_dz_agent  = _mean_norm(agent_mask)

    # ── Linear reafference predictor: z_self + action → Δz_world ─────
    # Input: concat(z_self_prev [self_dim], a_prev [action_dim])
    # Output: Δz_world_vec [world_dim]
    # Train ONLY on empty-space steps, test on held-out 20% of empty steps

    r2_train = 0.0
    r2_test  = 0.0
    locomotion_explained_variance = 0.0

    if n_empty >= 20:
        X_all = torch.cat([z_self_all, a_all], dim=-1)  # [N, self_dim + action_dim]
        y_all = dz_vec_all                               # [N, world_dim]

        empty_indices = torch.where(empty_mask)[0]
        n_split = int(len(empty_indices) * 0.8)
        train_empty_idx = empty_indices[:n_split]
        test_empty_idx  = empty_indices[n_split:]

        train_mask_bool = torch.zeros(len(ttypes), dtype=torch.bool)
        test_mask_bool  = torch.zeros(len(ttypes), dtype=torch.bool)
        train_mask_bool[train_empty_idx] = True
        test_mask_bool[test_empty_idx]   = True

        r2_train, r2_test = _compute_r_squared(X_all, y_all, train_mask_bool, test_mask_bool)
        locomotion_explained_variance = max(0.0, r2_test)
        print(f"  Linear reafference predictor: R²_train={r2_train:.3f}  "
              f"R²_test={r2_test:.3f}  locomotion_explained_var={locomotion_explained_variance:.3f}",
              flush=True)
    else:
        print(f"  WARNING: only {n_empty} empty steps — skipping linear fit (need >= 20)",
              flush=True)

    # ── Perspective shift dominance ratio ─────────────────────────────
    # Ratio of empty-move Δz_world to env-caused Δz_world
    # > 1.0 means locomotion induces more z_world change than genuine world events (backwards!)
    if mean_dz_env > 1e-8:
        perspective_shift_dominance_ratio = mean_dz_empty / mean_dz_env
    else:
        perspective_shift_dominance_ratio = 0.0

    print(f"  mean_dz_world — empty: {mean_dz_empty:.4f}  "
          f"env_hazard: {mean_dz_env:.4f}  "
          f"dominance_ratio: {perspective_shift_dominance_ratio:.3f}", flush=True)

    # ── PASS / FAIL ───────────────────────────────────────────────────
    c1_pass = locomotion_explained_variance > 0.3
    c2_pass = mean_dz_env > mean_dz_empty
    c3_pass = n_empty >= 200
    c4_pass = True  # no fatal errors (we got here)
    fatal_errors = 0

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass
    status = "PASS" if all_pass else "FAIL"
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass])

    failure_notes = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: locomotion_explained_variance={locomotion_explained_variance:.3f} <= 0.3"
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: mean_dz_world_env={mean_dz_env:.4f} <= mean_dz_world_empty={mean_dz_empty:.4f}"
        )
    if not c3_pass:
        failure_notes.append(f"C3 FAIL: n_empty={n_empty} < 200")

    print(f"\nV3-EXQ-014 verdict: {status}  ({criteria_met}/4)", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)

    metrics = {
        "fatal_error_count":                   float(fatal_errors),
        "n_empty_moves":                       float(n_empty),
        "n_env_caused_hazard":                 float(n_env),
        "n_agent_caused_hazard":               float(n_agent),
        "mean_dz_world_empty":                 float(mean_dz_empty),
        "mean_dz_world_env_hazard":            float(mean_dz_env),
        "mean_dz_world_agent_hazard":          float(mean_dz_agent),
        "locomotion_explained_variance":       float(locomotion_explained_variance),
        "reafference_predictor_r2_train":      float(r2_train),
        "reafference_predictor_r2_test":       float(r2_test),
        "perspective_shift_dominance_ratio":   float(perspective_shift_dominance_ratio),
        "num_train_episodes":                  float(num_train_episodes),
        "crit1_pass": 1.0 if c1_pass else 0.0,
        "crit2_pass": 1.0 if c2_pass else 0.0,
        "crit3_pass": 1.0 if c3_pass else 0.0,
        "criteria_met": float(criteria_met),
    }

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(f"- {n}" for n in failure_notes)

    summary_markdown = f"""# V3-EXQ-014 — Perspective Shift Calibration

**Status:** {status}
**Training:** {num_train_episodes} eps (RANDOM policy, 12×12, 15 hazards, drift_interval=3, drift_prob=0.5)
**Eval:** {num_eval_episodes} eps
**Seed:** {seed}

## Motivation (SD-007 / MECH-098)

EXQ-012 revealed true calibration_gap ≈ 0.0007 — near-zero. Root cause: E2_world
takes an identity shortcut because the egocentric world_obs changes on every body
movement (perspective shift), not genuine world change. This experiment quantifies
that problem before implementing the reafference correction (SD-007).

## Mean ||Δz_world|| by Event Type

| Event Type | n | Mean Δz_world |
|---|---|---|
| empty_move (locomotion only) | {n_empty} | {mean_dz_empty:.4f} |
| env_caused_hazard (genuine world event) | {n_env} | {mean_dz_env:.4f} |
| agent_caused_hazard | {n_agent} | {mean_dz_agent:.4f} |

**Perspective shift dominance ratio:** {perspective_shift_dominance_ratio:.3f}
(ratio > 1.0 = locomotion induces more z_world change than genuine world events — the problem)

## Reafference Predictor (linear fit)

Linear predictor `Δz_world = W @ [z_self; action] + b` trained on empty-space steps:
- R² (train): {r2_train:.3f}
- R² (test, held-out empty steps): {r2_test:.3f}
- **Locomotion explained variance: {locomotion_explained_variance:.3f}**

High R² on held-out empty steps confirms that z_self + action can predict the
z_world change caused by locomotion — i.e., that z_world encodes perspective shift.
This is the perspective shift that SD-007 must subtract.

## PASS Criteria

| Criterion | Result | Value |
|---|---|---|
| C1: locomotion_explained_variance > 0.3 | {"PASS" if c1_pass else "FAIL"} | {locomotion_explained_variance:.3f} |
| C2: mean_dz_world_env_hazard > mean_dz_world_empty | {"PASS" if c2_pass else "FAIL"} | {mean_dz_env:.4f} vs {mean_dz_empty:.4f} |
| C3: n_empty >= 200 | {"PASS" if c3_pass else "FAIL"} | {n_empty} |
| C4: No fatal errors | {"PASS" if c4_pass else "FAIL"} | 0 |

Criteria met: {criteria_met}/4 → **{status}**
{failure_section}
"""

    return {
        "status": status,
        "metrics": metrics,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": "supports" if all_pass else ("mixed" if criteria_met >= 2 else "weakens"),
        "experiment_type": EXPERIMENT_TYPE,
        "fatal_error_count": fatal_errors,
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",           type=int, default=0)
    parser.add_argument("--train-episodes", type=int, default=300)
    parser.add_argument("--eval-episodes",  type=int, default=100)
    parser.add_argument("--steps",          type=int, default=200)
    args = parser.parse_args()

    result = run(
        seed=args.seed,
        num_train_episodes=args.train_episodes,
        num_eval_episodes=args.eval_episodes,
        steps_per_episode=args.steps,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"] = ts
    result["claim"] = CLAIM_IDS[0]
    result["verdict"] = result["status"]
    result["run_id"] = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = "ree_hybrid_guardrails_v1"

    out_dir = Path(__file__).resolve().parents[1] / "evidence" / "experiments" / EXPERIMENT_TYPE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{EXPERIMENT_TYPE}_{ts}.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"\nResult written to: {out_path}", flush=True)
    print(f"Status: {result['status']}", flush=True)
    for k, v in result["metrics"].items():
        print(f"  {k}: {v}", flush=True)
